import kbloch, use_autograd
import fft_funs as iff

if use_autograd.use == 0:
    # autograd turned off
    import numpy as np
    eig = np.linalg.eig
    inv = np.linalg.inv
    sqrt = np.sqrt
else:
    # autograd turned on
    import autograd.numpy as np
    from autograd import grad
    from autograd.extend import primitive, defvjp
    from primitives import vjp_maker_eig, vjp_maker_sqrt, vjp_maker_inv
    from primitives_fix import grad_eig
    import numpy as npf
    
    eig = primitive(npf.linalg.eig)
    sqrt = primitive(npf.sqrt)
    inv = primitive(npf.linalg.inv)
    #defvjp(eig, vjp_maker_eig)
    defvjp(eig, grad_eig)
    defvjp(sqrt, vjp_maker_sqrt)
    defvjp(inv, vjp_maker_inv)    

class RCWA_obj:
    def __init__(self,nG,L1,L2,freq,theta,phi,verbose=1):
        '''The time harmonic convention is exp(-i omega t), speed of light = 1
        The first and last layer must be uniform

        Two kinds of layers are currently supported: uniform layer,
        patterned layer from grids. Interface for patterned layer by
        direct analytic expression of Fourier series is included, but
        no examples inclded so far.

        nG: truncation order, but the actual truncation order might not be nG
        L1,L2: lattice vectors, in the list format, (x,y)

        '''
        # assert type(nG) == int, 'nG must be an integar'
        # assert type(theta) == float, 'angle theta should be a float'
        # assert type(phi) == float, 'angle phi should be a float'
        
        self.freq = freq
        self.omega = 2*np.pi*freq+0.j
        self.L1 = L1
        self.L2 = L2
        self.phi = phi
        self.theta = theta
        self.nG = nG
        self.verbose = verbose
        self.Layer_N = 0  # total number of layers
      
        # the length of the following variables = number of total layers
        self.thickness_list = []
        self.id_list = []  #[type, No., No. in patterned/uniform, No. in its family] starting from 0
        # type:0 for uniform, 1 for Grids, 2 for Fourier

        self.kp_list = []                
        self.q_list = []  # eigenvalues
        self.phi_list = [] #eigenvectors

        # Uniform layer
        self.Uniform_ep_list = []
        self.Uniform_N = 0
        
        # Patterned layer
        self.Patterned_N = 0  # total number of patterned layers        
        self.Patterned_epinv_list = []
        self.Patterned_ep2_list = []
        self.Patterned_epdiff_list = []
        self.Patterned_epbkg_list = []        
        
        # patterned layer from Grids
        self.GridLayer_N = 0
        self.GridLayer_Nxy_list = []

        # layers of analytic Fourier series (e.g. circles)
        self.FourierLayer_N = 0
        self.FourierLayer_params = []        
        
    def Add_LayerUniform(self,thickness,epsilon):
        #assert type(thickness) == float, 'thickness should be a float'

        self.id_list.append([0,self.Layer_N,self.Uniform_N])
        self.Uniform_ep_list.append(epsilon)
        self.thickness_list.append(thickness)
        
        self.Layer_N += 1
        self.Uniform_N += 1

    def Add_LayerGrid(self,thickness,epdiff,epbkg,Nx,Ny):
        self.thickness_list.append(thickness)
        self.Patterned_epdiff_list.append(epdiff)
        self.Patterned_epbkg_list.append(epbkg)
        self.GridLayer_Nxy_list.append([Nx,Ny])
        self.id_list.append([1,self.Layer_N,self.Patterned_N,self.GridLayer_N])

        self.Layer_N += 1
        self.GridLayer_N += 1
        self.Patterned_N += 1

    def Add_LayerFourier(self,thickness,epdiff,epbkg,params):
        self.thickness_list.append(thickness)
        self.Patterned_epdiff_list.append(epdiff)
        self.Patterned_epbkg_list.append(epbkg)
        self.FourierLayer_params.append(params)
        self.id_list.append([2,self.Layer_N,self.Patterned_N,self.FourierLayer_N])

        self.Layer_N += 1
        self.Patterned_N += 1
        self.FourierLayer_N += 1

    def Init_Setup(self,Gmethod=0):
        '''
        Set up reciprocal lattice (Gmethod:truncation scheme, 0 for circular, 1 for rectangular)
        Compute eigenvalues for uniform layers
        Initialize vectors for patterned layers
        '''
        kx0 = self.omega*np.sin(self.theta)*np.cos(self.phi)*sqrt(self.Uniform_ep_list[0])
        ky0 = self.omega*np.sin(self.theta)*np.sin(self.phi)*sqrt(self.Uniform_ep_list[0])

        # set up reciprocal lattice
        self.Lk1, self.Lk2 = kbloch.Lattice_Reciprocate(self.L1,self.L2)
        self.G,self.nG = kbloch.Lattice_getG(self.nG,self.Lk1,self.Lk2,method=Gmethod)
        self.kx,self.ky = kbloch.Lattice_SetKs(self.G, kx0, ky0, self.Lk1, self.Lk2)
        
        #normalization factor for energies off normal incidence
        self.normalization = sqrt(self.Uniform_ep_list[0])/np.cos(self.theta)
        
        #if comm.rank == 0 and verbose>0:
        if self.verbose>0:
            print('Total nG = ',self.nG)

        self.Patterned_ep2_list = [None]*self.Patterned_N
        self.Patterned_epinv_list = [None]*self.Patterned_N            
        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega,0,1./ep,self.kx,self.ky)
                self.kp_list.append(kp)
                
                q,phi = SolveLayerEigensystem_uniform(self.omega,self.kx,self.ky,ep)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)
                
    def MakeExcitationPlanewave(self,p_amp,p_phase,s_amp,s_phase,order = 0):
        '''
        Front incidence
        '''
        theta = self.theta
        phi = self.phi
        a0 = np.zeros(2*self.nG,dtype=complex)
        bN = np.zeros(2*self.nG,dtype=complex)

        a0[order] = -s_amp*np.cos(theta)*np.cos(phi)*np.exp(1j*s_phase) \
            -p_amp*np.sin(phi)*np.exp(1j*p_phase)
        
        a0[order+self.nG] = -s_amp*np.cos(theta)*np.sin(phi)*np.exp(1j*s_phase) \
            +p_amp*np.cos(phi)*np.exp(1j*p_phase)
        
        self.a0 = a0
        self.bN = bN
        
    def GridLayer_getDOF(self,dof):
        '''
        Fourier transform + eigenvalue for grid layer
        '''
        ptri = 0
        ptr = 0
        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue
            
            Nx = self.GridLayer_Nxy_list[ptri][0]
            Ny = self.GridLayer_Nxy_list[ptri][1]
            dN = 1./Nx/Ny

            dofi = np.reshape(dof[ptr:ptr+Nx*Ny],[Nx,Ny])
            epdiff = self.Patterned_epdiff_list[self.id_list[i][2]]
            epbkg = self.Patterned_epbkg_list[self.id_list[i][2]]
            
            ep_grid = epdiff*dofi+epbkg
            epinv, ep2 = iff.GetEpsilon_FFT(dN,ep_grid,self.G)

            self.Patterned_epinv_list[self.id_list[i][2]] = epinv
            self.Patterned_ep2_list[self.id_list[i][2]] = ep2

            kp = MakeKPMatrix(self.omega,1,epinv,self.kx,self.ky)
            self.kp_list[self.id_list[i][1]] = kp

            q,phi = SolveLayerEigensystem(self.omega,self.kx,self.ky,kp,ep2)
            self.q_list[self.id_list[i][1]] = q
            self.phi_list[self.id_list[i][1]] = phi

            ptr += Nx*Ny
            ptri += 1

    def GridLayer_geteps(self,ep_all):
        '''
        Fourier transform + eigenvalue for grid layer
        '''
        ptri = 0
        ptr = 0
        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue
            
            Nx = self.GridLayer_Nxy_list[ptri][0]
            Ny = self.GridLayer_Nxy_list[ptri][1]
            dN = 1./Nx/Ny

            ep_grid = np.reshape(ep_all[ptr:ptr+Nx*Ny],[Nx,Ny])
            
            epinv, ep2 = iff.GetEpsilon_FFT(dN,ep_grid,self.G)

            self.Patterned_epinv_list[self.id_list[i][2]] = epinv
            self.Patterned_ep2_list[self.id_list[i][2]] = ep2

            kp = MakeKPMatrix(self.omega,1,epinv,self.kx,self.ky)
            self.kp_list[self.id_list[i][1]] = kp

            q,phi = SolveLayerEigensystem(self.omega,self.kx,self.ky,kp,ep2)
            self.q_list[self.id_list[i][1]] = q
            self.phi_list[self.id_list[i][1]] = phi

            ptr += Nx*Ny
            ptri += 1            

    def RT_Solve(self,normalize = 0):
        '''
        Reflection and transmission power computation
        Returns 2R and 2T, following Victor's notation
        Maybe because 2* makes S_z = 1 for H=1 in vacuum

        if normalize = 1, it will be divided by n[0]*cos(theta)
        '''
        aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)

        fi,bi = GetZPoyntingFlux(self.a0,b0,self.omega,self.kp_list[0],self.phi_list[0],self.q_list[0])
        fe,be = GetZPoyntingFlux(aN,self.bN,self.omega,self.kp_list[-1],self.phi_list[-1],self.q_list[-1])

        R = np.real(-bi)
        T = np.real(fe)

        if normalize == 1:
            R = R*self.normalization
            T = T*self.normalization
        return R,T

    def GetAmplitudes(self,which_layer,z_offset):
        '''
        returns fourier amplitude
        '''
        if which_layer == 0 :
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N:
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)

        ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],z_offset,ai,bi)

        return ai,bi
    
    def Solve_FieldFourier(self,which_layer,z_offset):
        '''
        returns field amplitude in fourier space: [ex,ey,ez], [hx,hy,hz]
        '''
        ai, bi = self.GetAmplitudes(which_layer,z_offset)

        # hx, hy in Fourier space
        fhxy = np.dot(self.phi_list[which_layer],ai+bi)
        fhx = fhxy[:self.nG]
        fhy = fhxy[self.nG:]

        # ex,ey in Fourier space
        tmp1 = (ai-bi)/self.omega/self.q_list[which_layer]
        tmp2 = np.dot(self.phi_list[which_layer],tmp1)
        fexy = np.dot(self.kp_list[which_layer],tmp2)
        fey = - fexy[:self.nG]
        fex = fexy[self.nG:]
        
        #hz in Fourier space
        fhz = (self.kx*fey - self.ky*fex)/self.omega

        #ez in Fourier space
        fez = (self.ky*fhx - self.kx*fhy)/self.omega
        if self.id_list[which_layer][0] == 0:
            fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            fez = np.dot(self.Patterned_epinv_list[self.id_list[which_layer][2]],fez)

        return [fex,fey,fez],[fhx,fhy,fhz]

    def Solve_FieldOnGrid(self,which_layer,z_offset):
        assert self.id_list[which_layer][0] == 1, 'Needs to be grids layer'

        Nxy = self.GridLayer_Nxy_list[self.id_list[which_layer][3]]
        Nx = Nxy[0]
        Ny = Nxy[1]

        # e,h in Fourier space
        fe,fh = self.Solve_FieldFourier(which_layer,z_offset)

        ex = iff.get_ifft(Nx,Ny,fe[0],self.G)
        ey = iff.get_ifft(Nx,Ny,fe[1],self.G)
        ez = iff.get_ifft(Nx,Ny,fe[2],self.G)

        hx = iff.get_ifft(Nx,Ny,fh[0],self.G)
        hy = iff.get_ifft(Nx,Ny,fh[1],self.G)
        hz = iff.get_ifft(Nx,Ny,fh[2],self.G)

        return [ex,ey,ez],[hx,hy,hz]

    def Volume_integral(self,which_layer,Mx,My,Mz,normalize=0):
        '''Mxyz is convolution matrix
        This function computes 1/A\int_V Mx|Ex|^2+My|Ey|^2+Mz|Ez|^2
        To be consistent with Poynting vector defintion here, the absorbed power will be just omega*output
        '''

        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]
        if self.id_list[which_layer][0] == 0:
            epinv = 1. / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        # amplitdue at z = 0 of that layer
        ai, bi = self.GetAmplitudes(which_layer,0.)
        ab = np.hstack((ai,bi))
        abMatrix = np.outer(np.conj(ab),ab)
        
        Mt = Matrix_zintegral(q,self.thickness_list[which_layer])
        # overall
        abM = abMatrix * Mt

        # F matrix
        Faxy = np.dot(np.dot(kp,phi), np.diag(1./self.omega/q))
        Faz1 = 1./self.omega*np.dot(epinv,np.diag(self.ky))
        Faz2 = -1./self.omega*np.dot(epinv,np.diag(self.kx))
        Faz = np.dot(np.hstack((Faz1,Faz2)),phi)

        tmp1 = np.vstack((Faxy,Faz))
        tmp2 = np.vstack((-Faxy,Faz))
        F = np.hstack((tmp1,tmp2))

        # consider Mtotal
        Mzeros = np.zeros_like(Mx)
        Mtotal = np.vstack((np.hstack((Mx,Mzeros,Mzeros)),\
                            np.hstack((Mzeros,My,Mzeros)),\
                            np.hstack((Mzeros,Mzeros,Mz))))

        # integral = Tr[ abMatrix * F^\dagger *  Matconv *F ] 
        tmp = np.dot(np.dot(np.conj(np.transpose(F)),Mtotal),F)
        val = np.trace(np.dot(abM,tmp))

        if normalize == 1:
            val = val*self.normalization
        return val

    def Solve_ZStressTensorIntegral(self,which_layer):
        '''
        returns 2F_x,2F_y,2F_z, integrated over z-plane
        '''
        z_offset = 0.
        e,h = self.Solve_FieldFourier(which_layer,z_offset)
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        # compute D = epsilon E
        ## Dz = epsilon_z E_z = (ky*hx - kx*hy)/self.omega
        dz = (self.ky*hx - self.kx*hy)/self.omega

        ## Dxy = epsilon2 * Exy
        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = np.hstack((-ey,ex))
            dxy = np.dot(self.Patterned_ep2_list[self.id_list[which_layer][2]],exy)
            dx = dxy[self.nG:]
            dy = -dxy[:self.nG]

        Tx = np.sum(ex*np.conj(dz)+hx*np.conj(hz))
        Ty = np.sum(ey*np.conj(dz)+hy*np.conj(hz))
        Tz = 0.5*np.sum(ez*np.conj(dz)+hz*np.conj(hz)-ey*np.conj(dy)-ex*np.conj(dx)-np.abs(hx)**2-np.abs(hy)**2)

        Tx = np.real(Tx)
        Ty = np.real(Ty)
        Tz = np.real(Tz)

        return Tx,Ty,Tz

def MakeKPMatrix(omega,layer_type,epinv,kx,ky):
    nG = len(kx)
    
    # uniform layer, epinv has length 1
    if layer_type == 0:
        # JkkJT = np.block([[np.diag(ky*ky), np.diag(-ky*kx)],
        #                 [np.diag(-kx*ky),np.diag(kx*kx)]])

        Jk = np.vstack((np.diag(-ky),np.diag(kx)))
        JkkJT = np.dot(Jk,np.transpose(Jk))
        
        kp = omega**2*np.eye(2*nG) - epinv*JkkJT
    # patterned layer
    else:
        Jk = np.vstack((np.diag(-ky),np.diag(kx)))
        tmp = np.dot(Jk,epinv)
        kp = omega**2*np.eye(2*nG) - np.dot(tmp,np.transpose(Jk))
        
    return kp

def SolveLayerEigensystem_uniform(omega,kx,ky,epsilon):
    nG = len(kx)
    q = sqrt(epsilon*omega**2 - kx**2 - ky**2)
    # branch cut choice
    q = np.where(np.imag(q)<0.,-q,q)

    q = np.concatenate((q,q))
    phi = np.eye(2*nG)
    return q,phi

def SolveLayerEigensystem(omega,kx,ky,kp,ep2):
    nG = len(kx)
    
    #k = np.block([ [np.diag(kx)],[np.diag(ky)] ])
    k = np.vstack((np.diag(kx),np.diag(ky)))
    kkT = np.dot(k,np.transpose(k))
    M = np.dot(ep2,kp) - kkT
    
    q,phi = eig(M)

    q = sqrt(q)
    # branch cut choice
    q = np.where(np.imag(q)<0.,-q,q)
    return q,phi

def GetSMatrix(indi,indj,q_list,phi_list,kp_list,thickness_list):
    ''' S_ij: size 4n*4n
    '''
    assert type(indi) == int, 'layer index i must be integar'
    assert type(indj) == int, 'layer index j must be integar'
    
    nG2 = len(q_list[0])
    S11 = np.eye(nG2,dtype=complex)
    S12 = np.zeros_like(S11)
    S21 = np.zeros_like(S11)
    S22 = np.eye(nG2,dtype=complex)
    if indi == indj:
        return S11,S12,S21,S22
    elif indi>indj:
        raise Exception('indi must be < indj')
   
    for l in range(indi,indj):
        ## next layer
        lp1 = l+1

        ## Q = inv(phi_l) * phi_lp1
        Q = np.dot(inv(phi_list[l]),  phi_list[lp1])
        ## P = ql*inv(kp_l*phi_l) * kp_lp1*phi_lp1*q_lp1^-1
        P1 = np.dot(np.diag(q_list[l]),   inv(np.dot(kp_list[l],phi_list[l])))
        P2 = np.dot(np.dot(kp_list[lp1],phi_list[lp1]),   np.diag(1./q_list[lp1]))
        P = np.dot(P1,P2)
        # P1 = np.dot(kp_list[l],phi_list[l])
        # P2 = np.dot(np.dot(kp_list[lp1],phi_list[lp1]),   np.diag(1./q_list[lp1]))
        # P = np.linalg.solve(P1,P2)
        # P = np.dot(np.diag(q_list[l]),P)

        #T11=T22, T12=T21
        T11 = 0.5*(Q+P)
        T12 = 0.5*(Q-P)

        # phase
	d1 = np.diag(np.exp(1j*q_list[l]*thickness_list[l]))
        d2 = np.diag(np.exp(1j*q_list[lp1]*thickness_list[lp1]))

        # S11 = inv(T11-d1*S12o*T12)*d1*S11o
        P1 = T11 - np.dot(np.dot(d1,S12),T12)
        P1 = inv(P1)  # hold for further use
        S11 = np.dot(np.dot(P1,d1),S11)

        # S12 = P1*(d1*S12o*T11-T12)*d2
        P2 = np.dot(d1,np.dot(S12,T11))-T12
        S12 = np.dot(np.dot(P1,P2),d2)

        # S21 = S22o*T12*S11+S21o
        S21 = S21 + np.dot(S22,np.dot(T12,S11))

        # S22 = S22o*T12*S12+S22o*T11*d2
        P2 = np.dot(S22,np.dot(T12,S12))
        P1 = np.dot(S22,np.dot(T11,d2))
        S22 = P1 + P2
        
    return S11,S12,S21,S22

def SolveExterior(a0,bN,q_list,phi_list,kp_list,thickness_list):
    '''
    Given a0, bN, solve for b0, aN
    '''

    Nlayer = len(thickness_list) # total number of layers
    S11, S12, S21, S22 = GetSMatrix(0,Nlayer-1,q_list,phi_list,kp_list,thickness_list)

    aN = np.dot(S11,a0) + np.dot(S12,bN)
    b0 = np.dot(S21,a0) + np.dot(S22,bN)

    return aN,b0

def SolveInterior(which_layer,a0,bN,q_list,phi_list,kp_list,thickness_list):
    '''
    Given a0, bN, solve for ai, bi
    Layer numbering starts from 0
    '''

    Nlayer = len(thickness_list) # total number of layers
    nG2 = len(q_list[0])
    
    S11, S12, S21, S22 = GetSMatrix(0,which_layer,q_list,phi_list,kp_list,thickness_list)
    pS11, pS12, pS21, pS22 = GetSMatrix(which_layer,Nlayer-1,q_list,phi_list,kp_list,thickness_list)

    # tmp = inv(1-S12*pS21)
    tmp = inv(np.eye(nG2)-np.dot(S12,pS21))
    # ai = tmp * (S11 a0 + S12 pS22 bN)
    ai = np.dot(tmp,  np.dot(S11,a0)+np.dot(S12,np.dot(pS22,bN)))
    # bi = pS21 ai + pS22 bN
    bi = np.dot(pS21,ai) + np.dot(pS22,bN)
    
    return ai,bi

def  TranslateAmplitudes(q,thickness,dz,ai,bi):
    ai = ai*np.exp(1j*q*dz)
    bi = bi*np.exp(1j*q*(thickness-dz))
    return ai,bi
        

def GetZPoyntingFlux(ai,bi,omega,kp,phi,q):
    '''
     Returns 2S_z/A, following Victor's notation
     Maybe because 2* makes S_z = 1 for H=1 in vacuum
    '''
    # A = kp phi inv(omega*q)
    A = np.dot(np.dot(kp,phi),  np.diag(1./omega/q))

    pa = np.dot(phi,ai)
    pb = np.dot(phi,bi)
    Aa = np.dot(A,ai)
    Ab = np.dot(A,bi)

    # diff = 0.5*(pb* Aa - Ab* pa)
    diff = 0.5*np.sum(np.conj(pb)*Aa-np.conj(Ab)*pa)
    #forward = real(Aa* pa) + diff
    forward = np.real(np.sum(np.conj(Aa)*pa)) + diff
    backward = -np.real(np.sum(np.conj(Ab)*pb)) + np.conj(diff)

    return forward, backward
    #return np.real(forward), np.real(backward)

def Matrix_zintegral(q,thickness):
    ''' Generate matrix for z-integral
    '''
    nG2 = len(q)
    qi,qj = np.meshgrid(q,q,indexing='ij')

    # # Maa = \int exp(i q_i z)^* exp(i q_j z)
    # #     = [exp(i(q_j-q_i^*)t)-1]/i(q_j-q_i^*)
    # # Mbb = \int exp(i q_i (t-z))^* exp(i q_j (t-z))
    # #     = exp(i(q_j-q_i^*)t)  [exp(i(q_i^*-q_j)t)-1]/i(q_i^*-q_j)
    # #     = Maa
    # # Mab = \int exp(i q_i z)^* exp(i q_j (t-z))
    # #     = exp(iq_j t) [1-exp(-i(q_i^*+q_j)t)]/i(q_i^*+q_j)
    # #     = [exp(iq_j t)-exp(-i q_i^* t)]/i(q_i^*+q_j)
    # # Mba = \int exp(i q_i (t-z))^* exp(i q_j z)
    # #     = exp(-iq_i^* t) [exp(i(q_j+q_i^*)t)-1]/i(q_j+q_i^*)
    # #     = Mab
    # Maa = (np.exp(1j*(qj-np.conj(qi))*thickness)-1)/1j/(qj-np.conj(qi))
    # Mab = (np.exp(1j*qj*thickness)-np.exp(-1j*np.conj(qi)*thickness))/1j/(qj+np.conj(qi))

    # M = t exp(0.5it (qj-qi^*)) sinc(0.5d (sjqj-siqi^*), note in python sinc = sin(pi x)/pi x
    qij = qj-np.conj(qi)
    Maa = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij/np.pi)

    qij2 = qj+np.conj(qi)
    Mab = thickness * np.exp(0.5j*thickness*qij) * np.sinc(0.5*thickness*qij2/np.pi)

    tmp1 = np.vstack((Maa,Mab))
    tmp2 = np.vstack((Mab,Maa))
    Mt = np.hstack((tmp1,tmp2))
    return Mt

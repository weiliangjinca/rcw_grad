import use_autograd

if use_autograd.use == 0:
    import numpy as np
    import numpy as npf
    inv = np.linalg.inv
else:
    import autograd.numpy as np
    from autograd import grad
    from autograd.extend import primitive, defvjp
    from primitives import vjp_maker_inv
    import numpy as npf

    inv = primitive(npf.linalg.inv)
    defvjp(inv, vjp_maker_inv)

def GetEpsilon_FFT(dN,eps_grid,G):
    '''dN = 1/Nx/Ny
    eps_grid is a numpy 2d array in the format of (Nx,Ny)
    
    For now, assume epsilon is isotropic
    if epsilon has xz,yz component, just simply add them to off-diagonal eps2
    '''

    eps_fft = get_conv(dN,eps_grid,G)
    epsinv = inv(eps_fft)
    # somehow block don't work with autograd
    # eps2 = np.block([[eps_fft,np.zeros_like(eps_fft)],
    #                  [np.zeros_like(eps_fft),eps_fft]])
    
    tmp1 = np.vstack((eps_fft,np.zeros_like(eps_fft)))
    tmp2 = np.vstack((np.zeros_like(eps_fft),eps_fft))
    eps2 = np.hstack((tmp1,tmp2))

    return epsinv, eps2
    
def get_conv(dN,s_in,G):
    ''' Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    nG,_ = G.shape
    sfft = np.fft.fft2(s_in)*dN
    
    # s_out = np.zeros((nG,nG),dtype=complex)
    # for i in range(nG):
    #     for j in range(nG):
    #         s_out[i, j] = sfft[G[i,0]-G[j,0], G[i,1]-G[j,1]]

    ix = range(nG)
    ii,jj = np.meshgrid(ix,ix,indexing='ij')
    s_out = sfft[G[ii,0]-G[jj,0], G[ii,1]-G[jj,1]]    
    return s_out

def get_fft(dN,s_in,G):
    '''
    FFT to get Fourier components
    
    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    
    sfft = np.fft.fft2(s_in)*dN
    # s_out = np.zeros(nG,dtype=complex)
    # for i in range(nG):
    #     s_out[i] = sfft[G[i,0],G[i,1]]
    return sfft[G[:,0],G[:,1]]

def get_ifft(Nx,Ny,s_in,G):
    '''
    Reconstruct real-space fields
    '''
    dN = 1./Nx/Ny
    nG,_ = G.shape

    s0 = np.zeros((Nx,Ny),dtype=complex)
    for i in range(nG):
        x = G[i,0]
        y = G[i,1]
        #if x>-Nx/2 and x<Nx/2 and y>-Ny/2 and y<Ny/2
        s0[x,y] = s_in[i]

    s_out = np.fft.ifft2(s0)/dN
    return s_out

# def circle_in_square_fft(Lx,Lk1,Lk2,radius,epbkg,epdiff,G):
#     from scipy.special import j1
#     nG,_ = G.shape

#     u = np.linalg.norm(Lk1)
#     v = np.linalg.norm(Lk2)
#     uv = np.dot(Lk1,Lk2)    

#     s_out = npf.zeros((nG,nG),dtype=complex)
#     for i in range(nG):
#         for j in range(nG):
#             if i == j:
#                 val = epbkg + epdiff*np.pi*(radius/Lx)**2
#             else:
#                 G1 = G[i,0]-G[j,0]
#                 G2 = G[i,1]-G[j,1]
#                 Gr = radius*2*np.pi*np.sqrt(G1**2*u**2+G2**2*v**2+2*G2*G1*uv)

#                 val = np.pi*(radius/Lx)**2*epdiff*j1(Gr)/Gr
#                 val *= 2  # why?
#             s_out[i,j]=val
#     return s_out

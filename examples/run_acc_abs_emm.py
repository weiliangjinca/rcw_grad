import os,sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # for hera
sys.path.append('/home/wljin/MyLocal/RCWA/')
sys.path.append('/home/wljin/MyLocal/RCWA/examples/')
sys.path.append('/home/wljin/MyLocal/RCWA/materials/')

import autograd.numpy as np
import numpy as npf
from mpi4py import MPI

import use_autograd
use_autograd.use = 1
import rcwa
import materials, cons
from mpi_nlopt import nlopt_opt,b_filter,f_symmetry
from fft_funs import get_conv
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
##### input parameters
Nlayer=2
lam0=1.2e-6
Qref = 20
Qabs = 1e10
c_ref = 30.
c_abs = 1000.
nG = 51
bproj = 0.
init_type = 'vac'
thickness = [150e-9,150e-9]
materialL = ['silicon','silica']
Period = 3e-6
xsym = 1
ysym = 1
Mx = 50
My = 50 
# for thermal emission
T=625
lamsT=35e-6
lameT=7e-6
NfT = 2
NthetaT = 5
NphiT = 5
# pumping
Nf=20
mload = 1e-4
laserP = 1e10
area = 10
final_v = .2
epsimag = 6.6979e-07
#### process parameters
if xsym == 1:
    Nx = Mx*2
else:
    Nx=Mx
if ysym == 1:
    Ny = My*2
else:
    Ny=My

# materials
mstruct = []
for i in range(Nlayer):
    if materialL[i] == 'silica' and rank == 0:
        mstruct.append(materials.silica())
    elif materialL[i] == 'silicon' and rank == 0:
        mstruct.append(materials.silicon())
mstruct = comm.bcast(mstruct)

# thermal emission
prefactorT = 4*0.5 # 4 for symmetry, 0.5 for polarization
fs=lam0/lamsT
fe=lam0/lameT
dfreq = (fe-fs)/NfT
dtheta = np.pi/2/NthetaT
dphi = np.pi/2/NphiT
freq0 = np.linspace(fs+dfreq/2,fe-dfreq/2,NfT)
phi0 = np.linspace(dphi/2,np.pi/2-dphi/2,NphiT)
theta0 = np.linspace(dtheta/2,np.pi/2-dtheta/2,NthetaT)
Ntotal = NfT*NphiT*NthetaT
freqL,phiL,thetaL = np.meshgrid(freq0,phi0,theta0,indexing='ij')

# pumping
beta = np.linspace(0,final_v,Nf)
freq_list = np.sqrt((1-beta)/(1+beta))
gamma = 1./np.sqrt(1-beta**2)

filename = './DATA/acc_all'+'_N'+str(Nlayer)+'_sym'+str(xsym)+str(ysym)+'_consref'+str(c_ref)+'_consabs'+str(c_abs)+'_bproj'+str(bproj)+'_nG'+str(nG)+'_Pmicron'+str(Period*1e6)+'_Nf'+str(Nf)+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_Q'+str(Qref)+'_T'+str(T)+'_NfT'+str(NfT)+'_Ntheta'+str(NthetaT)+'_'
for i in range(Nlayer):
    filename += materialL[i]+'_tnm'+str(thickness[i]*1e9)+'_'

# start to assemble RCWA
# lattice vector
Lx = Period/lam0
Ly = Period/lam0
L1 = [Lx,0.]
L2 = [0.,Ly]

# now consider four layers: vacuum + patterned + vacuum
epsuniform = 1.
epsbkg = 1

thick0 = 1.
thick = [ f/lam0 for f in thickness]
thickN = 1.

def rcwa_assembly(dofold,freq,theta,phi,planewave):
    '''
    planewave:{'p_amp',...}
    '''
    df = f_symmetry(dofold,Mx,My,xsym,ysym,Nlayer=Nlayer)
    dof = b_filter(df,bproj)

    obj = rcwa.RCWA_obj(nG,L1,L2,freq,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick0,epsuniform)
    epsdiff=[]
    for i in range(Nlayer):
        epsdiff.append(mstruct[i].epsilon(lam0/np.real(freq),x_type = 'lambda')-epsbkg)
        obj.Add_LayerGrid(thick[i],epsdiff[i],epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniform)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_getDOF(dof)
    
    return obj,dof,epsdiff

def accelerate_D(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    # RCWA
    freqcmp = freq_list[ctrl]*(1+1j/2/Qref)
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    theta = 0.
    phi = 0.
    obj,dof,epsdiff = rcwa_assembly(dofold,freqcmp,theta,phi,planewave)
    R,_ = obj.RT_Solve(normalize=1)

    # mass
    mT = mload
    for i in range(Nlayer):
        mT = mT + thickness[i]*area*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])

    dbeta = beta[1]-beta[0]
    integrand = mT/R*gamma[ctrl]*beta[ctrl]/(1-beta[ctrl])**2
    integrand = cons.c**3/2/laserP/area*integrand*dbeta/1e9

    # trapz rule for even sampling
    if ctrl == 0 or ctrl == Nf-1:
        integrand = 0.5*integrand
    return integrand

def infoR(dofold,val):
    df = f_symmetry(dofold,Mx,My,xsym,ysym,Nlayer=Nlayer)
    dof = b_filter(df,bproj)
    mT = mload
    for i in range(Nlayer):
        mT = mT + thickness[i]*area*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])
    F = np.mean(gamma*beta*cons.c/(1-beta)**2)
    R = 1./(1e9*val*2*laserP*area/cons.c**2/mT/F/beta[-1])
    return (R,np.mean(dof))

def p_abs(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    # RCWA
    freqcmp = freq_list[ctrl]*(1+1j/2/Qabs)
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    theta = 0.
    phi = 0.
    obj,dof,epsdiff = rcwa_assembly(dofold,freqcmp,theta,phi,planewave)
    # volume integration
    vals = 0.0
    for i in range(Nlayer):
        Mv=get_conv(1./Nx/Ny,dof[i*Nx*Ny:(i+1)*Nx*Ny].reshape((Nx,Ny)),obj.G)
        #vals = vals + np.real(obj.Volume_integral(1+i,Mv,Mv,Mv,normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])
        vals = vals + np.real(obj.Volume_integral(1+i,Mv,Mv,Mv,normalize=1))*np.real(obj.omega)*epsimag

    vals = vals*laserP
    vals = vals * (1-beta[ctrl])/(1+beta[ctrl]) # relativistic correction

    return vals

def emissivity_f(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    average over p/s polarizations, sum over forward and backward directions
    '''
    freq = freqL.flatten()[ctrl]
    theta = thetaL.flatten()[ctrl]
    phi = phiL.flatten()[ctrl]
    # RCWA
    freqcmp = freq*(1+1j/2/Qabs)
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj,dof,epsdiff = rcwa_assembly(dofold,freqcmp,theta,phi,planewave)

    Mv = []
    for i in range(Nlayer):
        Mv.append(get_conv(1./Nx/Ny,dof[i*Nx*Ny:(i+1)*Nx*Ny].reshape((Nx,Ny)),obj.G))

    # planewave excitation
    p_phase = 0.
    s_phase = 0.
    p_amp = 0.
    s_amp = 1.
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0,direction='forward')
    vals = 0.0
    for i in range(Nlayer):
        vals = vals + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0,direction='backward')
    for i in range(Nlayer):
        vals = vals + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])

    p_amp = 1.
    s_amp = 0.
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0,direction='forward')
    valp = 0.0
    for i in range(Nlayer):
        valp = valp + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0,direction='backward')
    for i in range(Nlayer):
        valp = valp + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])

    val = (vals+valp)*np.sin(theta)*np.cos(theta)/np.pi
    val = val * dtheta*dphi*prefactorT

    return val

def emission_integrate(dofold,ctrl):
    freq = freqL.flatten()[ctrl]

    omegaSI = 2*np.pi*freq*cons.c/lam0
    domega = 2*np.pi*dfreq*cons.c/lam0
    theta = cons.hbar*omegaSI/(np.exp(cons.hbar*omegaSI/cons.k/T)-1)

    val = emissivity_f(dofold,ctrl)*domega
    phi = val/(2*np.pi)**2*(omegaSI/cons.c)**2*theta
    return phi

# # nlopt setup
ndof = Mx*My*Nlayer

ismax = 1 # 0 for minimization
lb = 0.
ub = 1.
maxeval = 500
ftol = 1e-10
savefile_N = 2


obj = [emission_integrate,Ntotal,'sum']
constraint=[[[accelerate_D,c_ref],Nf,'sum'],[[p_abs,c_abs],Nf,'logsumexp']]
nopt = nlopt_opt(ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,info=['cons1','  (R,V) = ',infoR],xsym=xsym,ysym=ysym,bproj=bproj)
x = nopt.fun_opt(ismax,obj,init_type,constraint=constraint)


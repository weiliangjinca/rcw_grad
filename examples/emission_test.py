import os,sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # for hera
sys.path.append('/home/wljin/MyLocal/RCWA/')
sys.path.append('/home/wljin/MyLocal/RCWA/examples/')

import autograd.numpy as np
import numpy as npf

import use_autograd
use_autograd.use = 1
import rcwa
from mpi_nlopt import nlopt_opt,b_filter,f_symmetry
from fft_funs import get_conv
import cons

# input parameters
direction = 'forward'
Qabs = 1e10
nG = 21
init_type = 'one'
Nlayer = 2
epsdiff = [10+1j,10+1j]
thickness = [150e-9,150e-9]
Period = 3e-6

lam0=1e-6
T=1000
lams=35e-6
lame=7e-6
Nf = 10
Ntheta = 5
Nphi = 5
prefactor = 4*0.5 # 4 for symmetry, 0.5 for polarization

fs=lam0/lams
fe=lam0/lame
dfreq = (fe-fs)/Nf
dtheta = np.pi/2/Ntheta
dphi = np.pi/2/Nphi
freq0 = np.linspace(fs+dfreq/2,fe-dfreq/2,Nf)
phi0 = np.linspace(dphi/2,np.pi/2-dphi/2,Nphi)
theta0 = np.linspace(dtheta/2,np.pi/2-dtheta/2,Ntheta)

Ntotal = Nf*Nphi*Ntheta
freqL,phiL,thetaL = np.meshgrid(freq0,phi0,theta0,indexing='ij')

bproj = 0.
xsym = 1
ysym = 1
Mx = 30
My = 30 
if xsym == 1:
    Nx = Mx*2
else:
    Nx=Mx
if ysym == 1:
    Ny = My*2
else:
    Ny=My

# start to assemble RCWA
# lattice vector
Lx = Period/lam0
Ly = Period/lam0
L1 = [Lx,0.]
L2 = [0.,Ly]

# now consider three layers: vacuum + patterned + vacuum
epsuniform = 1.
epsbkg = 1

thick0 = 1.
thick = [ f/lam0 for f in thickness]
thickN = 1.

def emissivity_f(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    df = f_symmetry(dofold,Mx,My,xsym,ysym,Nlayer=Nlayer)
    dof = b_filter(df,bproj)

    freq = freqL.flatten()[ctrl]
    theta = thetaL.flatten()[ctrl]
    phi = phiL.flatten()[ctrl]

    freqcmp = freq*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick0,epsuniform)
    for i in range(Nlayer):
        obj.Add_LayerGrid(thick[i],epsdiff[i],epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniform)
    obj.Init_Setup(Gmethod=0)
    obj.GridLayer_getDOF(dof)

    Mv = []
    for i in range(Nlayer):
        Mv.append(get_conv(1./Nx/Ny,dof[i*Nx*Ny:(i+1)*Nx*Ny].reshape((Nx,Ny)),obj.G))

    # planewave excitation
    p_phase = 0.
    s_phase = 0.
    p_amp = 0.
    s_amp = 1.
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0,direction=direction)
    vals = 0.0
    for i in range(Nlayer):
        vals = vals + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])

    p_amp = 1.
    s_amp = 0.
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    valp = 0.0
    for i in range(Nlayer):
        valp = valp + np.real(obj.Volume_integral(1+i,Mv[i],Mv[i],Mv[i],normalize=1))*np.real(obj.omega)*np.imag(epsdiff[i])

    val = (vals+valp)*np.sin(theta)*np.cos(theta)/np.pi
    val = val * dtheta*dphi*prefactor

    return val

def emission_integrate(dofold,ctrl):
    freq = freqL.flatten()[ctrl]

    omegaSI = 2*np.pi*freq*cons.c/lam0
    domega = 2*np.pi*dfreq*cons.c/lam0
    theta = cons.hbar*omegaSI/(np.exp(cons.hbar*omegaSI/cons.k/T)-1)

    val = emissivity_f(dofold,ctrl)*domega
    phi = val/(2*np.pi)**2*(omegaSI/cons.c)**2*theta
    return phi

# nlopt setup
ndof = Mx*My*Nlayer

ismax = 0 # 0 for minimization
lb = 0.
ub = 1.
maxeval = 1
ftol = 1e-10
savefile_N = 5
filename='test.txt'

obj=[emission_integrate,Ntotal,'sum']
#obj=[emissivity_f,Ntotal,'sum']
nopt = nlopt_opt(ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,xsym=xsym,ysym=ysym,bproj=bproj)
x = nopt.fun_opt(ismax,obj,init_type)


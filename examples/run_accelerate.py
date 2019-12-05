import os,sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # for hera
sys.path.append('../')

import autograd.numpy as np
import numpy as npf

import use_autograd
use_autograd.use = 1
import rcwa, mpi_nlopt
c0 = 299792458.

# input parameters
Qabs = 20.
nG = 101
Nf = 40
Nx = 50
Ny = 50 
thickness = 100e-9
Period = 2e-6

lam0 = 1.2e-6
density = 2.329e3
epsdiff = 11.3

mload = 1e-4
laserP = 1e10
area = 10
final_v = 0.2

# doppler shift
v = np.linspace(0,final_v*c0,Nf)
freq_list = np.sqrt((c0-v)/(c0+v))
gamma = 1./np.sqrt(1-(v/c0)**2)

# start to assemble RCWA
# lattice vector
Lx = Period/lam0
Ly = Period/lam0
L1 = [Lx,0.]
L2 = [0.,Ly]

# planewave excitation
p_amp = 0.
s_amp = 1.
p_phase = 0.
s_phase = 0.
# normal incidence
theta = 0.
phi = 0.

# now consider three layers: vacuum + patterned + vacuum
epsuniform1 = 1.
epsbkg = 1
epsuniform3 = 1.

thick1 = 1.
thick2 = thickness/lam0
thick3 = 1.

def accelerate_D(dof,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    mT = mload + thickness*area*density*np.mean(dof)

    freqcmp = freq_list[ctrl]*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick1,epsuniform1)
    obj.Add_LayerGrid(thick2,epsdiff,epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thick3,epsuniform3)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    obj.GridLayer_getDOF(dof)
    R,_ = obj.RT_Solve()

    dv = (v[1]-v[0])
    integrand = mT/R*gamma[ctrl]*v[ctrl]/(1-v[ctrl]/c0)**2
    integrand = c0/2/laserP/area*integrand*dv/1e9

    # trapz rule for even sampling
    if ctrl == 0 or ctrl == Nf-1:
        integrand = 0.5*integrand
    return integrand


# nlopt setup
ndof = Nx*Ny
init = np.random.random(Nx*Ny)

ismax = 0 # 0 for minimization
lb = 0.
ub = 1.
maxeval = 100
xtol = 1e-5
filename = 'test'
savefile_N = 10

nopt = mpi_nlopt.nlopt_opt(ndof,lb,ub,maxeval,xtol,filename,savefile_N)
x = nopt.fun_opt(ismax,Nf,accelerate_D,init)

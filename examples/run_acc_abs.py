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
c0 = 299792458.

# input parameters
Qabs = 20.
init_type = 'vac'
material = 'Silicon'

bproj = 0.
xsym = 1
ysym = 1
nG = 101
Nf = 20
Mx = 50
My = 50 
if xsym == 1:
    Nx = Mx*2
else:
    Nx=Mx
if ysym == 1:
    Ny = My*2
else:
    Ny=My
thickness = 150e-9
Period = 3e-6
c_abs = 10.

lam0 = 1.2e-6
if material == 'Silicon':
    density = 2.329e3
    epsdiff = 11.3
elif material == 'test':
    density = 2.329e3
    epsdiff = 11.3+1j
elif material == 'Silver':
    density = 10.49e3
    epsdiff = -76+1.46j
else:
    raise Exception('Material not included')

mload = 1e-4
laserP = 1e10
area = 10
final_v = 0.2

filename = './DATA/acc_abs'+material+'_sym'+str(xsym)+str(ysym)+'_cons'+str(c_abs)+'_bproj'+str(bproj)+'_nG'+str(nG)+'_Pmicron'+str(Period*1e6)+'_tnm'+str(thickness*1e9)+'_Nf'+str(Nf)+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_Q'+str(Qabs)+'_'

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

def accelerate_D(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    df = f_symmetry(dofold,Mx,My,xsym,ysym)
    dof = b_filter(df,bproj)
    mT = mload + thickness*area*density*np.mean(dof)

    freqcmp = freq_list[ctrl]*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick1,epsuniform1)
    obj.Add_LayerGrid(thick2,epsdiff,epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thick3,epsuniform3)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    obj.GridLayer_getDOF(dof)
    R,_ = obj.RT_Solve(normalize=1)

    dv = (v[1]-v[0])
    integrand = mT/R*gamma[ctrl]*v[ctrl]/(1-v[ctrl]/c0)**2
    integrand = c0/2/laserP/area*integrand*dv/1e9

    # trapz rule for even sampling
    if ctrl == 0 or ctrl == Nf-1:
        integrand = 0.5*integrand
    return integrand

def infoR(dofold,val):
    df = f_symmetry(dofold,Mx,My,xsym,ysym)
    dof = b_filter(df,bproj)
    mT = mload + thickness*area*density*np.mean(dof)
    F = np.mean(gamma*v/(1-v/c0)**2)
    R = 1./(1e9*val*2*laserP*area/c0/mT/F/v[-1])
    return (R,np.mean(dof))

def p_abs(dofold,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    df = f_symmetry(dofold,Mx,My,xsym,ysym)
    dof = b_filter(df,bproj)

    freqcmp = freq_list[ctrl]*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick1,epsuniform1)
    obj.Add_LayerGrid(thick2,epsdiff,epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thick3,epsuniform3)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    obj.GridLayer_getDOF(dof)

    Mv = get_conv(1./Nx/Ny,dof.reshape((Nx,Ny)),obj.G)
    val = obj.Volume_integral(1,Mv,Mv,Mv,normalize=1)

    integrand = np.real(val)/Nf

    # trapz rule for even sampling
    if ctrl == 0 or ctrl == Nf-1:
        integrand = 0.5*integrand
    return integrand

# nlopt setup
ndof = Mx*My

ismax = 0 # 0 for minimization
lb = 0.
ub = 1.
maxeval = 500
ftol = 1e-10
savefile_N = 10

obj = [p_abs,Nf]
constraint=[[accelerate_D,c_abs],Nf]
nopt = nlopt_opt(ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,info=['cons','  (R,V) = ',infoR],xsym=xsym,ysym=ysym,bproj=bproj)
x = nopt.fun_opt(ismax,obj,init_type,constraint=constraint)


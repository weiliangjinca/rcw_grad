import os,sys, argparse
Nthread = 1
os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

rpath = '/home/asgard/rcw_grad/'
sys.path.append(rpath)
sys.path.append(rpath+'examples/')
sys.path.append(rpath+'materials/')

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
parser = argparse.ArgumentParser()
# solvers
parser.add_argument('-nG', action="store", type=int, default=101)
parser.add_argument('-Nlayer', action="store", type=int, default=1)
parser.add_argument('-thickness', action="store", type=float, default=1)
parser.add_argument('-bproj', action="store", type=float, default=0)
parser.add_argument('-Qref', action="store", type=float, default=1e10)
parser.add_argument('-mload', action="store", type=float, default=1e-5)
parser.add_argument('-Period', action="store", type=float, default=1e-6)
parser.add_argument('-init_type', action="store", type=str, default='vac')

r, unknown = parser.parse_known_args(sys.argv[1:])
if rank == 0:
    for arg in vars(r):
        print(arg," is ",getattr(r,arg))

nG = r.nG
init_type = r.init_type
bproj = r.bproj
Qref = r.Qref
mload = r.mload

Nlayer=r.Nlayer
thickness = [r.thickness/Nlayer]*Nlayer
materialL = ['silicon']*Nlayer
lam0 = 1.2e-6

Period = r.Period
xsym = 0
ysym = 0
Mx = 100
My = 100 

# pumping
Nf=24
final_v = .2
laserP = 1e10
epsimag = 0.
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
        mstruct.append(materials.silicon(epsimag = epsimag))
    elif materialL[i] == 'gold' and rank == 0:
        mstruct.append(materials.gold())
mstruct = comm.bcast(mstruct)

# pumping
dbeta = (final_v-0)/Nf
beta = np.linspace(0+dbeta/2,final_v-dbeta/2,Nf)
freq_list = np.sqrt((1-beta)/(1+beta))
gamma = 1./np.sqrt(1-beta**2)

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
    rho = mload
    for i in range(Nlayer):
        rho = rho + thickness[i]*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])

    integrand = rho/R*gamma[ctrl]*beta[ctrl]/(1-beta[ctrl])**2
    integrand = cons.c**3/2/laserP*integrand*dbeta/1e9

    return integrand

def infoR(dofold,val):
    df = f_symmetry(dofold,Mx,My,xsym,ysym,Nlayer=Nlayer)
    dof = b_filter(df,bproj)
    rho = mload
    for i in range(Nlayer):
        rho = rho + thickness[i]*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])

    F = np.mean(gamma*beta/(1-beta)**2)

    R = 1./(1e9*val*2*laserP/cons.c**3/rho/F/beta[-1])
    return (R,np.mean(dof),(rho-mload)/mload)

# # nlopt setup
ndof = Mx*My*Nlayer

lb = 0.
ub = 1.
maxeval = 500
ftol = 1e-10
savefile_N = 2

ismax = 0 # 0 for minimization
obj = [accelerate_D,Nf,'sum']

filename = './DATA/acc'+'_N'+str(Nlayer)+'_sym'+str(xsym)+str(ysym)+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_Pmicron'+str(Period*1e6)+'_mload'+str(mload)+'_Nf'+str(Nf)+'_Q'+str(Qref)+'_nG'+str(nG)+'_bproj'+str(bproj)+'_'
for i in range(Nlayer):
    filename += materialL[i]+'_tnm'+str(thickness[i]*1e9)+'_'

nopt = nlopt_opt(ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,info=['obj','  (R,V,Mratio) = ',infoR],xsym=xsym,ysym=ysym,bproj=bproj,Nlayer=Nlayer)
x = nopt.fun_opt(ismax,obj,init_type)

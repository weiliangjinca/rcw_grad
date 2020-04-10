import os,sys, argparse
Nthread = 1
os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

rpath = '/home/wljin/MyLocal/RCWA/'
sys.path.append(rpath)
sys.path.append(rpath+'examples/')
sys.path.append(rpath+'materials/')

import autograd.numpy as np
from autograd.scipy.special import logsumexp
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
parser.add_argument('-Job', action="store", type=int, default=1)
parser.add_argument('-topt', action="store", type=int, default=0)
parser.add_argument('-material', action="store", type=str, default='silicon')
parser.add_argument('-inverse', action="store", type=int, default=0)
parser.add_argument('-Mx', action="store", type=int, default=100)
parser.add_argument('-dx', action="store", type=float, default=1e-4)
parser.add_argument('-ind', action="store", type=int, default=0)
parser.add_argument('-nG', action="store", type=int, default=101)
parser.add_argument('-Nlayer', action="store", type=int, default=1)
parser.add_argument('-thickness', action="store", type=float, default=1)
parser.add_argument('-angle', action="store", type=float, default=0.)
parser.add_argument('-bproj', action="store", type=float, default=0)
parser.add_argument('-Qref', action="store", type=float, default=1e10)
parser.add_argument('-mload', action="store", type=float, default=0.1)
parser.add_argument('-mpower', action="store", type=float, default=1.0)
parser.add_argument('-Period', action="store", type=float, default=1e-6)
parser.add_argument('-init_type', action="store", type=str, default='vac')
parser.add_argument('-polarization', action="store", type=str, default='s')

r, unknown = parser.parse_known_args(sys.argv[1:])
if rank == 0:
    for arg in vars(r):
        print(arg," is ",getattr(r,arg))

nG = r.nG
init_type = r.init_type
bproj = r.bproj
Qref = r.Qref
mload = r.mload*1e-4 # 1e-3 for g to kg, and 1e-1 for 10m^2

Nlayer=r.Nlayer
thickness = [r.thickness/Nlayer]*Nlayer
materialL = [r.material]*Nlayer
lam0 = 1.2e-6

Period = r.Period
xsym = 0
ysym = 0
Mx = r.Mx
My = r.Mx

# pumping
Nf=20
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

if init_type == 'F':
    filling = 0.2
    dinit = []
    if r.topt == 1:
        dinit = [np.array([0.5])]
    for i in range(Nlayer):
        dtmp = np.zeros((Nx,Ny),dtype=float)
        nx = int(filling*Nx)
        if i%2 == 1 and (r.polarization == 'ps' or r.polarization == 'sp'):
            dtmp[:,:nx] = 1.
        else:
            dtmp[:nx,:] = 1.
        dinit.append(dtmp.flatten())
    dinit = np.concatenate(np.array(dinit))
    npf.savetxt('./DATA/tmp.txt', dinit)
    init_type = './DATA/tmp.txt'
    
# materials
mstruct = []
for i in range(Nlayer):
    if materialL[i] == 'silica' and rank == 0:
        mstruct.append(materials.silica())
    elif materialL[i] == 'silicon' and rank == 0:
        mstruct.append(materials.silicon(epsimag = epsimag))
    elif materialL[i] == 'SiN' and rank == 0:
        mstruct.append(materials.SiN(epsimag = epsimag))
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

def rcwa_assembly(dofold,freq,theta,phi,planewave,pthick):
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
        obj.Add_LayerGrid(pthick[i],epsdiff[i],epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniform)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_getDOF(dof)
    
    return obj,dof,epsdiff

def accelerate_D(doftotal,ctrl):
    ''' ctrl: ctrl's frequency calculation
    '''
    if r.topt == 1:
        pthick = [thick[i]*doftotal[i] for i in range(Nlayer)]
        dofold = doftotal[Nlayer:]
    else:
        pthick = thick
        dofold = doftotal

    # RCWA
    freqcmp = freq_list[ctrl]*(1+1j/2/Qref)
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    phi = 0.
    theta = r.angle
    obj,dof,epsdiff = rcwa_assembly(dofold,freqcmp,theta,phi,planewave,pthick)
    R,_ = obj.RT_Solve(normalize=1)
    
    if r.polarization == 'ps' or r.polarization == 'sp':
        planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
        R2,_ = obj.RT_Solve(normalize=1)

        # the minimal of reflection
        Inc = 500.
        R = Inc/logsumexp(Inc/np.array([R,R2]))

    # mass
    rho = mload
    for i in range(Nlayer):
        mtmp = lam0*pthick[i]*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])
        rho = rho + mtmp
    rho = rho**r.mpower

    integrand = rho/R*gamma[ctrl]*beta[ctrl]/(1-beta[ctrl])**2
    integrand = cons.c**3/2/laserP*integrand*dbeta/1e9

    return integrand

def infoR(doftotal,val):
    if r.topt == 1:
        pthick = [thick[i]*doftotal[i] for i in range(Nlayer)]
        dofold = doftotal[Nlayer:]
    else:
        pthick = thick
        dofold = doftotal

    df = f_symmetry(dofold,Mx,My,xsym,ysym,Nlayer=Nlayer)
    dof = b_filter(df,bproj)

    if r.inverse == 1:
        val = 1/val
    rho = mload
    rhor = mload
    for i in range(Nlayer):
        mtmp = lam0*pthick[i]*mstruct[i].density*np.mean(dof[i*Nx*Ny:(i+1)*Nx*Ny])
        rho = rho + mtmp
        rhor = rhor + mtmp
    rho = rho**r.mpower
    F = np.mean(gamma*beta/(1-beta)**2)

    R = 1./(1e9*val*2*laserP/cons.c**3/rho/F/beta[-1])
    return (R,np.mean(dof),(rhor-mload)/mload,[pthick[i]*lam0*1e9 for i in range(Nlayer)])

# # nlopt setup
ndof = Mx*My*Nlayer
if r.topt == 1:
    ndof = ndof + Nlayer

lb = 0.
ub = 1.
maxeval = 500
ftol = 1e-10
savefile_N = 2

ismax = 0 # 0 for minimization
obj = [accelerate_D,Nf,'sum']

filename = './DATA/acc'+'_topt'+str(r.topt)+'_inv'+str(r.inverse)+'_N'+str(Nlayer)+'_'+r.polarization+'_sym'+str(xsym)+str(ysym)+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_Pmicron'+str(Period*1e6)+'_mload'+str(r.mload)+'_Nf'+str(Nf)+'_Qf'+str(Qref)+'_angle'+str(r.angle)+'_nG'+str(nG)+'_bproj'+str(bproj)+'_mload'+str(mload*1e4)+'_mp'+str(r.mpower)+'_'
for i in range(Nlayer):
    filename += materialL[i]+'_tnm'+str(thickness[i]*1e9)+'_'

nopt = nlopt_opt(ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,info=['obj','  (R,V,Mratio,t) = ',infoR],xsym=xsym,ysym=ysym,bproj=bproj,Nlayer=Nlayer)

if r.Job == 1:
    x = nopt.fun_opt(ismax,obj,init_type,inverse=r.inverse)
elif r.Job == 0:
    nopt.fun_testgrad(obj,init_type,r.dx,r.ind)

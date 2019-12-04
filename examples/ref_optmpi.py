import os,sys
os.environ["OPENBLAS_NUM_THREADS"] = "1" # for hera
sys.path.append('/home/wljin/MyLocal/RCWA/')

import autograd.numpy as np
from autograd import grad
import nlopt, time, numpy as npf
from mpi4py import MPI
import use_autograd
use_autograd.use = 1
import rcwa

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t = 59e-9
Period = 1.24e-6

c0 = 299792458
rho = 2.329e3
mp = 1e-4;
I = 1e10;
A = 10
vf = 0.2*c0
lam0 = 1.2e-6

N = 40
v = np.linspace(0,vf,N)
freq_list = np.sqrt((c0-v)/(c0+v))
gamma = 1./np.sqrt(1-(v/c0)**2)

Nx = 50
Ny = 50

nG = 101
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

# frequency and angles
theta = 0.
phi = 0.

# now consider three layers: vacuum + patterned + vacuum
epsuniform1 = 1.
epsdiff = 11.3
epsbkg = 1
epsuniform3 = 1.

thick1 = 1.
thick2 = t/lam0
thick3 = 1.

def fun_freq(dof,ind,Qabs):
    mT = mp + t*A*rho*np.mean(dof)
    freqcmp = freq_list[ind]*(1+1j/2/Qabs)

    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick1,epsuniform1)
    obj.Add_LayerGrid(thick2,epsdiff,epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thick3,epsuniform3)
    obj.Init_Setup(Gmethod=0)
    
    obj.MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase,order = 0)
    obj.GridLayer_getDOF(dof)
    R,_ = obj.RT_Solve()

    integrand = mT/R*gamma[ind]*v[ind]/(1-v[ind]/c0)**2
    D = c0/2/I/A*integrand*(v[1]-v[0])/1e9
    if ind == 0 or ind == N-1:
        D = 0.5*D

    return D


def fun_mpi(dof,Qabs):
    dof = comm.bcast(dof)

    Nloop = int(np.ceil(1.0*N/size)) # number of calculations for each node
    Dl=[]
    gl=[]
    D=[]
    gradn=[]

    for i in range(0,Nloop):
        ind = i*size+rank
        if ind < N:
            fun = lambda dof: fun_freq(dof,ind,Qabs)
            grad_fun = grad(fun)

            val = fun(dof)
            gval = grad_fun(dof)

            Dl.append([ind,val])
            gl.append([ind,gval])

    Dl = comm.gather(Dl)
    gl = comm.gather(gl)

    if rank == 0:
        Dl = npf.concatenate(npf.array(Dl))
        gl = npf.concatenate(npf.array(gl))
        # sindex = Dl[:,0].argsort()

        # Dl = Dl[sindex,1]
        # gl = gl[sindex,1]

        D = np.sum(Dl[:,1])
        gradn = np.sum(gl[:,1])

    D = comm.bcast(D)
    gradn = comm.bcast(gradn)
    return D,gradn

Qabs = 20.
def fun_nlopt(dof,gradn):
    D,gn = fun_mpi(dof,Qabs)
    gradn[:] = gn

    if 'autograd' not in str(type(D)) and rank == 0:
        print D
    return D

ndof = Nx*Ny
lb=np.zeros(ndof,dtype=float)
ub=np.ones(ndof,dtype=float)

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(100)

opt.set_min_objective(fun_nlopt)
x = opt.optimize(np.random.random(ndof))


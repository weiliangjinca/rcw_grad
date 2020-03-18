import os,sys,time
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

import numpy as np
from mpi4py import MPI

import use_autograd
use_autograd.use = 0
import rcwa
import materials, cons
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

nG = 301
Qref = 1e10
Nfreq=161
NF=50

freqL = np.linspace(0.2,1.8,Nfreq)
FL = np.linspace(0.02,1.,NF)

Nlayer=1
thickness = [150e-9]
materialL = ['silicon']
lam0 = 1.2e-6
Period = 1e-6

print_n=50
name_ = 'nG'+str(nG)+'_thick'+str(thickness[0]*1e9)+'_P'+str(Period*1e6)+'.txt'
xsym = 0
ysym = 0
Mx = 100
My = 100 
epsimag = 0.
Nx=Mx
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

def rcwa_assembly(filling,freq,theta,phi,planewave):
    '''
    planewave:{'p_amp',...}
    '''
    dof = np.zeros((Nx,Ny),dtype=float)
    nx = int(filling*Nx)
    dof[:nx,:] = 1.
    
    obj = rcwa.RCWA_obj(nG,L1,L2,freq,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick0,epsuniform)
    epsdiff=[]
    for i in range(Nlayer):
        epsdiff.append(mstruct[i].epsilon(lam0/np.real(freq),x_type = 'lambda')-epsbkg)
        obj.Add_LayerGrid(thick[i],epsdiff[i],epsbkg,Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniform)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_getDOF(dof.flatten())
    
    return obj

    
Ntotal = Nfreq*NF
Nloop = int(np.ceil(1.0*Ntotal/size)) # number of calculations for each node

RL=[]
time0 = time.time()
for i in range(0,Nloop):
    ind = i*size+rank
    if ind < Ntotal:
        iF = int(ind % NF)
        ifreq = int(ind / NF)

        freq = freqL[ifreq]
        F = FL[iF]
            
        freqcmp = freq*(1+1j/2/Qref)
        planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
        theta = 0.
        phi = 0.
        obj = rcwa_assembly(F,freqcmp,theta,phi,planewave)
        R,_ = obj.RT_Solve(normalize=1)

        RL.append([freq,F,R])
                
        if rank == 0 and i%print_n == 0:
            time2 = time.time()
            print('-- Round ',i,' finished, out of ',Nloop, ' , time elasped = ',time2-time0)
        
RL = comm.gather(RL)
if rank == 0:
    RL = np.concatenate(np.array(RL))
    RL = RL[RL[:,0].argsort()] # sort along omega

    for i in range(0,Nfreq):
        RLt = RL[i*NF:(i+1)*NF,:]
        RL[i*NF:(i+1)*NF,:] = RLt[RLt[:,1].argsort()] # sort along kx

    np.savetxt(name_,RL)        

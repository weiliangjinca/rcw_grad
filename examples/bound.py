import os, time, argparse
# Nthread = 5
# os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

import sys
sys.path.append('/home1/06500/tg857995/rcw_grad')
sys.path.append('/home1/06500/tg857995/rcw_grad/materials/')
sys.path.append('/home1/06500/tg857995/rcw_grad/examples/')

import autograd.numpy as np
from autograd import grad
import nlopt, numpy as npf
from scipy.optimize import fsolve as solve

import use_autograd
use_autograd.use = 1
import rcwa
import materials, cons
from fft_funs import get_conv

parser = argparse.ArgumentParser()
# solvers
parser.add_argument('-nG', action="store", type=int, default=101)
parser.add_argument('-Nlayer', action="store", type=int, default=1)
parser.add_argument('-thick', action="store", type=float, default=1)
parser.add_argument('-Lx', action="store", type=float, default=1)
parser.add_argument('-bproj', action="store", type=float, default=0)
parser.add_argument('-epreal', action="store", type=float, default=16)
parser.add_argument('-epimag', action="store", type=float, default=0.1)
parser.add_argument('-init_type', action="store", type=str, default='vac')

r, unknown = parser.parse_known_args(sys.argv[1:])
for arg in vars(r):
    print(arg," is ",getattr(r,arg))
        
epsdiff = r.epreal+r.epimag*1j
nG = r.nG 
bproj = r.bproj
Nlayer=r.Nlayer
thick = r.thick
Lx = r.Lx
Ly = Lx
freq = 1.
Mx = 100
My = 100
Qabs = 1e20

name_ = './DATA/abs_N'+str(Nlayer)+'_nG'+str(nG)+'_t'+str(thick)+'_L'+str(Lx)+'_epr'+str(r.epreal)+'_epi'+str(r.epimag)+'_b'+str(bproj)

ndof = Mx*My*Nlayer
thickness = [thick/Nlayer]*Nlayer

L1 = [Lx,0.]
L2 = [0.,Ly]
epsuniform = 1.
epsbkg = 1.

thick0 = 1.
thickN = 1.

if r.init_type == 'rand':
    init = np.random.random(ndof)
elif r.init_type == 'vac':
    init = np.zeros(ndof)+1e-3*np.random.random(ndof)
elif r.init_type == 'one':
    init = np.ones(ndof)
else:
    tmp = open(init_type,'r')
    init = np.loadtxt(tmp)

def fun_owen(ep):
    sigma = np.imag(ep)/np.abs(ep-1)**2
    fun = lambda h:h-2/np.pi*sigma/(1-np.sinc(2*h)**2)
    init = 1/np.imag(np.sqrt(ep))/2/np.pi
    return solve(fun,init)[0]

print('epsilon=',epsdiff+epsbkg,'skin deptph=',1/np.imag(np.sqrt(epsdiff+epsbkg))/2/np.pi,'owen=',fun_owen(epsdiff+epsbkg))

def b_filter(dof,bproj):
    eta = 0.5
    dofnew = np.where(dof<=eta,eta*(np.exp(-bproj*(1-dof/eta))-(1-dof/eta)*np.exp(-bproj)),(1-eta)*(1-np.exp(-bproj*(dof-eta)/(1-eta)) + (dof - eta)/(1-eta) * np.exp(-bproj)) + eta)
    return dofnew

def rcwa_assembly(dof,nG,bproj,theta,phi,planewave):
    '''
    planewave:{'p_amp',...}
    '''
    freqcmp = freq*(1+1j/2/Qabs)
    obj = rcwa.RCWA_obj(nG,L1,L2,freqcmp,theta,phi,verbose=0)
    obj.Add_LayerUniform(thick0,epsuniform)
    for i in range(Nlayer):
        obj.Add_LayerGrid(thickness[i],epsdiff,epsbkg,Mx,My)
    obj.Add_LayerUniform(thickN,epsuniform)
    obj.Init_Setup(Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_getDOF(dof)

    return obj

ctrl = 0
def p_abs(dofold,nG,bproj,method='RT'):
    dof = b_filter(dofold,bproj)
    
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    theta = 0.
    phi = 0.
    
    t1 = time.time()
    obj= rcwa_assembly(dof,nG,bproj,theta,phi,planewave)

    vals = 0.
    R = 0.
    T = 0.
    
    if method == 'V':
        for i in range(Nlayer):
            Mv=get_conv(1./Mx/My,dof[i*Mx*My:(i+1)*Mx*My].reshape((Mx,My)),obj.G)
            vals = vals + np.real(obj.Volume_integral(1+i,Mv,Mv,Mv,normalize=1))*np.real(obj.omega)*np.imag(epsdiff)
    elif method == 'RT':
        R,T = obj.RT_Solve(normalize=1) 
        vals = 1-R-T
    t2 = time.time()
    
    if 'autograd' not in str(type(vals)):
        global ctrl
        if npf.mod(ctrl,2) == 0:
            npf.savetxt(name_+'_dof'+str(ctrl)+'.txt', dof)        
        if ctrl<=1:
            print(t2-t1)
        if method == 'V':
            print(ctrl,vals)
        elif method == 'RT':
            print(ctrl, 'R =',R,'T =',T,'Abs=',vals)
            
        ctrl +=1
    return vals

method = 'RT'
fun = lambda dof: p_abs(dof,nG,bproj,method = method)
grad_fun = grad(fun)
def fun_nlopt(dof,gradn):
    gradn[:] = grad_fun(dof)
    return fun(dof)

lb=np.zeros(ndof,dtype=float)
ub=np.ones(ndof,dtype=float)

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(1000)

opt.set_max_objective(fun_nlopt)
x = opt.optimize(init)

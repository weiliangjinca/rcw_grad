import os, time
Nthread = 5
os.environ["OMP_NUM_THREADS"] = str(Nthread) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(Nthread) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(Nthread) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(Nthread) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(Nthread) # export NUMEXPR_NUM_THREADS=1

import sys
sys.path.append('/home/weiliang/Local/rcw_grad/')
sys.path.append('/home/weiliang/Local/rcw_grad/materials/')
sys.path.append('/home/weiliang/Local/rcw_grad/examples/')

import autograd.numpy as np
from autograd import grad
import nlopt, numpy as npf
from scipy.optimize import fsolve as solve

import use_autograd
use_autograd.use = 1
import rcwa
import materials, cons
from fft_funs import get_conv

lam0=0.8e-6
freq = 1.
gold = materials.gold()
silicon = materials.silicon()
Nlayer=1
thick =.004
Lx = 5
Ly = Lx
Mx = 100
My = 100
Qabs = 1e10

ndof = Mx*My*Nlayer
thickness = [thick/Nlayer]*Nlayer

L1 = [Lx,0.]
L2 = [0.,Ly]
epsuniform = 1.
epsbkg = 1.
epsdiff = silicon.epsilon(lam0/freq,'lambda')-epsbkg

thick0 = 1.
thickN = 1.

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
        np.savetxt('dof.txt',dof)
        if ctrl<=1:
            print(t2-t1)
        if method == 'V':
            print(ctrl,vals)
        elif method == 'RT':
            print(ctrl, 'R =',R,'T =',T,'Abs=',vals,'time=',t2-t1)
            
        ctrl +=1
    return vals

nG = 201
bproj = 0.
method = 'RT'
fun = lambda dof: p_abs(dof,nG,bproj,method = method)
grad_fun = grad(fun)
def fun_nlopt(dof,gradn):
    gradn[:] = grad_fun(dof)
    return fun(dof)

init = np.random.random(ndof)
#init = np.zeros(ndof)+1e-3*np.random.random(ndof)
init = np.ones(ndof)
#tmp = open('./dof.txt','r')
#init = np.loadtxt(tmp)
lb=np.zeros(ndof,dtype=float)
ub=np.ones(ndof,dtype=float)

opt = nlopt.opt(nlopt.LD_MMA, ndof)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.set_xtol_rel(1e-5)
opt.set_maxeval(3000)

opt.set_max_objective(fun_nlopt)
x = opt.optimize(init)

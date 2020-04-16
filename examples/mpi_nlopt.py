from mpi4py import MPI
import autograd.numpy as np
from autograd import grad
from scipy.special import logsumexp
import nlopt, numpy as npf
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class nlopt_opt:
    def __init__(self,ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,bproj=0.,xsym=0,ysym=0,info=[None],Nlayer=1,timing=1):
        '''
        savefile_N: output dof as file every such number, with filename
        '''
        if type(lb) == float or type(lb) == int:
            lbn = lb*np.ones(ndof,dtype=float)
        else:
            lbn = lb
        if type(ub) == float or type(ub) == int:
            ubn = ub*np.ones(ndof,dtype=float)
        else:
            ubn = ub

        opt = nlopt.opt(nlopt.LD_MMA, ndof)
        opt.set_lower_bounds(lbn)
        opt.set_upper_bounds(ubn)
        opt.set_maxeval(maxeval)
        opt.set_ftol_rel(ftol)
        self.opt = opt

        self.Mx = Mx
        self.My = My
        self.filename = filename
        self.savefile_N = savefile_N
        self.info = info
        self.ndof = ndof
        self.ctrl = 0
        self.bproj = bproj
        self.xsym = xsym
        self.ysym = ysym
        self.Nlayer = Nlayer
        self.timing = timing

    def fun_testgrad(self,fun,init_type,dx,ind):
        init = []
        if rank == 0:
            init = get_init(init_type,self.ndof)
        init = comm.bcast(init)

        val1,gn1 = fun_mpi(init,fun[0],fun[1],output=fun[2])
        
        init[ind] += dx        
        val2,gn2 = fun_mpi(init,fun[0],fun[1],output=fun[2])

        if rank == 0:
            print('Finite difference = ',(val2-val1)/dx, ', Auto = ',(gn1[ind]+gn2[ind])/2)        
        
    def fun_opt(self,ismax,fun,init_type,constraint=None,inverse=0):
        '''
        fun[0]:fun(dof,ctrl): function that returns integrand at ctrl's frequency
        fun[1]:N: number of parallel frequency computations
        fun[2]:output_type
        constraint=[[cons_fun,cons_max],N,output_type]
        '''
        init = []
        if rank == 0:
            init = get_init(init_type,self.ndof)
        init = comm.bcast(init)

        def fun_nlopt(dof,gradn):
            t1 = time.time()
            val0,gn = fun_mpi(dof,fun[0],fun[1],output=fun[2])
            if inverse == 1:
                val = 1./val0
                gn = -gn*val**2
            else:
                val = val0
            gradn[:] = gn
            t2 = time.time()

            if 'autograd' not in str(type(val)) and rank == 0:
                if self.timing == 1:
                    print(self.ctrl,'val = ',val0,'time=',t2-t1)
                else:
                    print(self.ctrl,'val = ',val0)
                if self.info[0] == 'obj':
                    R = self.info[2](dof,val)
                    print('   ',self.info[1],R)
            
                if self.savefile_N>0 and npf.mod(self.ctrl,self.savefile_N) == 0:
                    npf.savetxt(self.filename+'dof'+str(self.ctrl)+'.txt', dof)
                    if self.bproj>0 or self.xsym==1 or self.ysym==1:
                        df = f_symmetry(dof,self.Mx,self.My,self.xsym,self.ysym,Nlayer=self.Nlayer)
                        dofnew = b_filter(df,self.bproj)
                        npf.savetxt(self.filename+'doftrans'+str(self.ctrl)+'.txt', dofnew)

            self.ctrl += 1
            return val

        if constraint != None and type(constraint[1])!=list:
            def fun_cons(dof,gradn):
                val,gn = fun_mpi(dof,constraint[0][0],constraint[1],output=constraint[2])
                gradn[:] = gn

                if 'autograd' not in str(type(val)) and rank == 0:
                    print(self.ctrl,'cons = ',val)

                    if self.info[0] == 'cons':
                        R = self.info[2](dof,val)
                        print('   ',self.info[1],R)

                return val-constraint[0][1]
            self.opt.add_inequality_constraint(fun_cons, 1e-8)

        if constraint != None and type(constraint[1])==list:
            def fun_cons1(dof,gradn):
                val,gn = fun_mpi(dof,constraint[0][0][0],constraint[0][1],output=constraint[0][2])
                gradn[:] = gn

                if 'autograd' not in str(type(val)) and rank == 0:
                    print(self.ctrl,'cons = ',val)

                    if self.info[0] == 'cons1':
                        R = self.info[2](dof,val)
                        print('   ',self.info[1],R)

                return val-constraint[0][0][1]

            def fun_cons2(dof,gradn):
                val,gn = fun_mpi(dof,constraint[1][0][0],constraint[1][1],output=constraint[1][2])
                gradn[:] = gn

                if 'autograd' not in str(type(val)) and rank == 0:
                    print(self.ctrl,'cons = ',val)

                    if self.info[0] == 'cons2':
                        R = self.info[2](dof,val)
                        print('   ',self.info[1],R)

                return val-constraint[1][0][1]

            self.opt.add_inequality_constraint(fun_cons1, 1e-8)
            self.opt.add_inequality_constraint(fun_cons2, 1e-8)

        if (ismax == 1 and inverse == 0) or (ismax == 0 and inverse == 1):
            self.opt.set_max_objective(fun_nlopt)
        else:
            self.opt.set_min_objective(fun_nlopt)

        x = self.opt.optimize(init)
        return x

def b_filter(dof,bproj):
    eta = 0.5
    dofnew = np.where(dof<=eta,eta*(np.exp(-bproj*(1-dof/eta))-(1-dof/eta)*np.exp(-bproj)),(1-eta)*(1-np.exp(-bproj*(dof-eta)/(1-eta)) + (dof - eta)/(1-eta) * np.exp(-bproj)) + eta)
    return dofnew

def f_symmetry(dof,Mx,My,xsym,ysym,Nlayer=1):
    out = []
    for i in range(Nlayer):
        df = np.reshape(dof[i*Mx*My:(i+1)*Mx*My],(Mx,My))
        if xsym == 1:
            df = np.hstack((df,np.fliplr(df)))
        if ysym == 1:
            df = np.vstack((df,np.flipud(df)))
        out.append(df.flatten())
    return np.concatenate(np.array(out))

def fun_mpi(dof,fun,N,output='sum'):
    '''mpi parallization for fun(dof,ctrl), ctrl is the numbering of ctrl's frequency calculation
    N calculations in total
    returns the sum: sum_{ctrl=1 toN} fun(dof,ctrl)
    '''
    dof = comm.bcast(dof)

    Nloop = int(np.ceil(1.0*N/size)) # number of calculations for each node
    val_i=[]
    g_i=[]
    val=[]
    g=[]

    for i in range(0,Nloop):
        ctrl = i*size+rank
        if ctrl < N:

            funi = lambda dof: fun(dof,ctrl)
            grad_fun = grad(funi)

            val = funi(dof)
            gval = grad_fun(dof)

            # include indexing for now, in case one is interested
            val_i.append([ctrl,val])
            g_i.append([ctrl,gval])

    # gather the solution
    val_i = comm.gather(val_i)
    g_i = comm.gather(g_i)

    # summation
    if rank == 0:
        val_i = [x for x in val_i if x]
        g_i = [x for x in g_i if x]
        val_i = npf.concatenate(npf.array(val_i))
        g_i = npf.concatenate(npf.array(g_i))
        # sindex = val_i[:,0].argsort()

        # val_i = val_i[sindex,1]
        # g_i = g_i[sindex,1]

        if output == 'sum':
            val = np.sum(val_i[:,1])
            g = np.sum(g_i[:,1])
        elif output == 'logsumexp':
            val = logsumexp(val_i[:,1])
            g = np.zeros_like(g_i[0,1])
            for i in range(N):
                g += g_i[i,1]*np.exp(val_i[i,1]-val)

    val = comm.bcast(val)
    g = comm.bcast(g)
    return val,g

def fun_ratio(dof,fun1,fun2,N1,N2,output1='sum',output2='sum'):
    '''
    y = fun1(x)/fun2(x)
    '''
    val1,grad1 = fun_mpi(dof,fun1,N1,output1)
    val2,grad2 = fun_mpi(dof,fun2,N2,output2)
    
    val = val1/val2
    grad = (grad1*val2-grad2*val1)/val2**2
    return val,grad

def get_init(init_type,ndof):
    if init_type == 'rand':
        init = np.random.random(ndof)
    elif init_type == 'vac':
        init = np.zeros(ndof)+1e-5*np.random.random(ndof)
    elif init_type == 'one':
        init = np.ones(ndof)
    else:
        tmp = open(init_type,'r')
        initfile = np.loadtxt(tmp)

        if len(initfile) == ndof:
            init = initfile+np.random.random(ndof)*1e-2
            init = init - np.random.random(ndof)*1e-2
            init[init>1.]=1.
            init[init<0.]=0.
        elif len(initfile) < ndof:
            init = np.zeros(ndof,dtype=float)
            init[:len(initfile)]=initfile

    return init

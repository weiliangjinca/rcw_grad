from mpi4py import MPI
import autograd.numpy as np
from autograd import grad
from scipy.special import logsumexp
import nlopt, numpy as npf
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class nlopt_opt:
    def __init__(self,ndof,lb,ub,maxeval,ftol,filename,savefile_N,Mx,My,bproj=0.,xsym=0,ysym=0,info=None):
        '''
        savefile_N: output dof as file every such number, with filename
        '''
        lbn=lb*np.ones(ndof,dtype=float)
        ubn=ub*np.ones(ndof,dtype=float)

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

    def fun_opt(self,ismax,fun,init_type,constraint=None):
        '''
        fun[0]:fun(dof,ctrl): function that returns integrand at ctrl's frequency
        fun[1]:N: number of parallel frequency computations
        fun[2]:output_type
        constraint=[[cons_fun,cons_max],N,output_type]
        '''
        init = []
        if rank == 0:
            if init_type == 'rand':
                init = np.random.random(self.ndof)
            elif init_type == 'vac':
                init = np.zeros(self.ndof)+1e-5*np.random.random(self.ndof)
            elif init_type == 'one':
                init = np.ones(self.ndof)
            else:
                tmp = open(init_type,'r')
                init = np.loadtxt(tmp)
        init = comm.bcast(init)

        def fun_nlopt(dof,gradn):
            val,gn = fun_mpi(dof,fun[0],fun[1],output=fun[2])
            gradn[:] = gn

            if 'autograd' not in str(type(val)) and rank == 0:
                print self.ctrl,'val = ',val
                if self.info[0] == 'obj':
                    R = self.info[2](dof,val)
                    print '   ',self.info[1],R
            
                if self.savefile_N>0 and npf.mod(self.ctrl,self.savefile_N) == 0:
                    npf.savetxt(self.filename+'dof'+str(self.ctrl)+'.txt', dof)
                    if self.bproj>0 or self.xsym==1 or self.ysym==1:
                        df = f_symmetry(dof,self.Mx,self.My,self.xsym,self.ysym)
                        dofnew = b_filter(df,self.bproj)
                        npf.savetxt(self.filename+'doftrans'+str(self.ctrl)+'.txt', dofnew)

            self.ctrl += 1
            return val

        if constraint != None:
            def fun_cons(dof,gradn):
                val,gn = fun_mpi(dof,constraint[0][0],constraint[1],output=constraint[2])
                gradn[:] = gn

                if 'autograd' not in str(type(val)) and rank == 0:
                    print self.ctrl,'cons = ',val

                    if self.info[0] == 'cons':
                        R = self.info[2](dof,val)
                        print '   ',self.info[1],R

                return val-constraint[0][1]

            self.opt.add_inequality_constraint(fun_cons, 1e-8)

        if ismax == 1:
            self.opt.set_max_objective(fun_nlopt)
        else:
            self.opt.set_min_objective(fun_nlopt)
            
        x = self.opt.optimize(init)
        return x

def b_filter(dof,bproj):
    eta = 0.5
    dofnew = np.where(dof<=eta,eta*(np.exp(-bproj*(1-dof/eta))-(1-dof/eta)*np.exp(-bproj)),(1-eta)*(1-np.exp(-bproj*(dof-eta)/(1-eta)) + (dof - eta)/(1-eta) * np.exp(-bproj)) + eta)
    return dofnew

def f_symmetry(dof,Mx,My,xsym,ysym):
    df = np.reshape(dof,(Mx,My))
    if xsym == 1:
        df = np.hstack((df,np.fliplr(df)))
    if ysym == 1:
        df = np.vstack((df,np.flipud(df)))
    return df.flatten()
        
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


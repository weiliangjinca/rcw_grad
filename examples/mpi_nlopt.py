from mpi4py import MPI
import autograd.numpy as np
from autograd import grad
import nlopt, numpy as npf
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class nlopt_opt:
    def __init__(self,ndof,lb,ub,maxeval,xtol,filename,savefile_N):
        '''
        savefile_N: output dof as file every such number, with filename
        '''
        lbn=lb*np.ones(ndof,dtype=float)
        ubn=ub*np.ones(ndof,dtype=float)

        opt = nlopt.opt(nlopt.LD_MMA, ndof)
        opt.set_lower_bounds(lbn)
        opt.set_upper_bounds(ubn)
        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(xtol)
        self.opt = opt

        self.filename = filename
        self.savefile_N = savefile_N
        self.ctrl = 0

    def fun_opt(self,ismax,N,fun,init):
        '''
        fun(dof,ctrl): function that returns integrand at ctrl's frequency
        N: number of parallel frequency computations
        '''
        def fun_nlopt(dof,gradn):
            val,gn = fun_mpi(dof,fun,N)
            gradn[:] = gn

            if 'autograd' not in str(type(val)) and rank == 0:
                print self.ctrl,val
            
                if self.savefile_N>0 and npf.mod(self.ctrl,self.savefile_N) == 0:
                    npf.savetxt(self.filename+'dof'+str(self.ctrl)+'.txt', dof)

            self.ctrl += 1
            return val

        if ismax == 1:
            self.opt.set_max_objective(fun_nlopt)
        else:
            self.opt.set_min_objective(fun_nlopt)

        x = self.opt.optimize(init)
        return x


def fun_mpi(dof,fun,N):
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

        val = np.sum(val_i[:,1])
        g = np.sum(g_i[:,1])

    val = comm.bcast(val)
    g = comm.bcast(g)
    return val,g


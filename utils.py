# import os, psutil,time
# from mpi4py.MPI import COMM_WORLD as comm


# def memory_report(ST=''):
#     process = psutil.Process(os.getpid())
#     h=process.memory_info().rss/1024/1024.0
#     print ST+ 'At core',str(comm.rank),' Used Memory = ' + str(h) +' MB'

def test_grad(fun,grad_fun,x,dx,ind):
    y1 = fun(x)
    x[ind] += dx
    y2 = fun(x)

    x[ind] -= dx/2
    g = grad_fun(x)
    print('Finite difference = ',(y2-y1)/dx, ', Auto = ',g[ind])

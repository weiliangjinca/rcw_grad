import numpy as np
from autograd.extend import primitive, defvjp, vspace

''' Define here various primitives needed for the main code 
To use with both numpy and autograd backends, define the autograd primitive of 
a numpy function fnc as fnc_ag, and then define the vjp'''

def T(x): return np.swapaxes(x, -1, -2)

'''=========== NP.SQRT STABLE AROUND 0 =========== '''
sqrt_ag = primitive(np.sqrt)

def vjp_maker_sqrt(ans, x):
    def vjp(g):
        return g * 0.5 * (x + 1e-10)**0.5/(x + 1e-10)
        # return np.where(np.abs(x) > 1e-10, g * 0.5 * x**-0.5, 0.)
    return vjp

defvjp(sqrt_ag, vjp_maker_sqrt)

def vjp_maker_meshgridx(ans, x):
    def vjp(g):
        return np.sum(g,axis=1)
    return vjp

'''=========== inv =========== '''

inv_ag = primitive(np.linalg.inv)

def vjp_maker_inv(ans, x):
    return lambda g: -np.dot(np.dot(T(ans), g), T(ans))
defvjp(inv_ag, vjp_maker_inv)

'''=========== NUMPY.LINALG.EIG =========== '''

eig = primitive(np.linalg.eig)

def vjp_maker_eig(ans, x):
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = np.transpose(np.linalg.inv(v))
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[:, np.newaxis], N, axis=-1)
        # Eigenvalue part
        vjp_temp = np.dot(vc * wg[np.newaxis, :], T(v)) 
        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function 
        # depends on the eigenvalues only.
        if np.any(vg):
            off_diag = np.ones((N, N)) - np.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + np.eye(N))
            vjp_temp += np.dot(np.dot(vc, F * np.dot(T(v), vg)), T(v))
        return vjp_temp
    return vjp
defvjp(eig, vjp_maker_eig)

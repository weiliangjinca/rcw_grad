import numpy as np

c0 = 299792458.
class silica:
    def __init__(self,filename='silica_data'):
        '''
        For fused quartz:
             dispersion (a-SiO2_llnl_cxro + a-SiO2_palik + Popova) from 1e-2 nm to 50 micron
        density is in unit of kg/m^3
        '''
        self.density = 2.203e3

        tmp = open(filename,'r')
        data = np.loadtxt(tmp)
        self.lam = data[:,0]
        self.n = data[:,1]
        self.k = data[:,2]

    def epsilon(self,x,x_type='lambda'):
        xlam = converter2lam(x,x_type)
    
        nout = np.interp(xlam,self.lam,self.n)
        kout = np.interp(xlam,self.lam,self.k)
        ep = (nout + 1j*kout)**2
        return ep

class silicon:
    def __init__(self,filename='si_data'):
        '''
        For crystalline silicon:
             dispersion (Si_llnl_cxro + Si_palik) from 1.2e-2 nm to 333 micron
        density is in unit of kg/m^3
        '''
        self.density = 2.329e3

        tmp = open(filename,'r')
        data = np.loadtxt(tmp)
        self.lam = data[:,0]/1e10
        self.n = data[:,1]
        self.k = data[:,2]

    def epsilon(self,x,x_type='lambda'):
        xlam = converter2lam(x,x_type)
    
        nout = np.interp(xlam,self.lam,self.n)
        kout = np.interp(xlam,self.lam,self.k)
        ep = (nout + 1j*kout)**2
        return ep    

def converter2lam(x,x_type):
    if x_type == 'lambda':
        return x
    elif x_type == 'freq':
        return c0/x
    elif x_type == 'omega':
        return 2*np.pi*c0/x

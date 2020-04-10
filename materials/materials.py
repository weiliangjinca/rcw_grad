import numpy as np

c0 = 299792458.
eV2freq= 2.417989261097518e+14

class SiN:
    def __init__(self,epsimag = 0.):
        '''
        0.31-5.5 micron, Luke 2015
        '''

        self.density = 3.17e3
        self.epsimag = epsimag

    def epsilon(self,x,x_type='lambda'):
        xlam = converter2lam(x,x_type)*1e6
        n = (1+3.0249/(1-(0.1353406/xlam)**2)+40314/(1-(1239.842/xlam)**2))**.5
        ep = n**2
        if np.imag(ep)<self.epsimag:
            ep = np.real(ep) + 1j * self.epsimag
        return ep

class silica:
    def __init__(self,filename='silica_data',epsimag = 0.):
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
        self.epsimag = epsimag

    def epsilon(self,x,x_type='lambda'):
        xlam = converter2lam(x,x_type)
    
        nout = np.interp(xlam,self.lam,self.n)
        kout = np.interp(xlam,self.lam,self.k)
        ep = (nout + 1j*kout)**2
        if np.imag(ep)<self.epsimag:
            ep = np.real(ep) + 1j * self.epsimag
        return ep

class silicon:
    def __init__(self,filename='si_data',epsimag = 0.):
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
        self.epsimag = epsimag

    def epsilon(self,x,x_type='lambda'):
        xlam = converter2lam(x,x_type)
    
        nout = np.interp(xlam,self.lam,self.n)
        kout = np.interp(xlam,self.lam,self.k)
        ep = (nout + 1j*kout)**2
        if np.imag(ep)<self.epsimag:
            ep = np.real(ep) + 1j * self.epsimag
        return ep    

class gold:
    def __init__(self):
        '''
        For gold, Lorentz-Drude model
        Original data: Rakic et al. 1998, https://doi.org/10.1364/AO.37.005271
        density is in unit of kg/m^3
        '''
        self.density = 19.3e3

    def epsilon(self,x,x_type='lambda'):
        xeV = converter2eV(x,x_type)
        
        wp = 9.03  #eV
        f0 = 0.760
        G0 = 0.053 #eV

        f1 = 0.024
        G1 = 0.241 #eV
        w1 = 0.415 #eV

        f2 = 0.010
        G2 = 0.345 #eV
        w2 = 0.830 #eV

        f3 = 0.071
        G3 = 0.870 #eV
        w3 = 2.969 #eV

        f4 = 0.601
        G4 = 2.494 #eV
        w4 = 4.304 #eV

        f5 = 4.384
        G5 = 2.214 #eV
        w5 = 13.32 #eV

        WP = f0**.5 * wp  #eV

        ep = 1-WP**2/(xeV*(xeV+1j*G0)) \
             + f1*wp**2 / ((w1**2-xeV**2)-1j*xeV*G1) \
             + f2*wp**2 / ((w2**2-xeV**2)-1j*xeV*G2) \
             + f3*wp**2 / ((w3**2-xeV**2)-1j*xeV*G3) \
             + f4*wp**2 / ((w4**2-xeV**2)-1j*xeV*G4) \
             + f5*wp**2 / ((w5**2-xeV**2)-1j*xeV*G5)

        return ep

def converter2lam(x,x_type):
    if x_type == 'lambda':
        return x
    elif x_type == 'freq':
        return c0/x
    elif x_type == 'omega':
        return 2*np.pi*c0/x

def converter2eV(x,x_type):
    if x_type == 'lambda':
        return c0/x/eV2freq
    elif x_type == 'freq':
        return x/eV2freq
    elif x_type == 'omega':
        return x/eV2freq/2/np.pi

from __future__ import print_function
import numpy as np
from .brent import brentq
from .constants import GasConst, sig_SB, AU, Omega0
from . import opacity
from .chambers_config import Config, Constants
################################################################################
# Thermodynamics classes
################################################################################
class EOS_Table(object):
    """Base class for equation of state evaluated at certain locations.

    Stores pre-computed temperatures, viscosities etc. Derived classes need to
    provide the funcitons called by set_grid.
    """
    def __init__(self):
        self._gamma = 1.0
        self._mu    = 2.4
    
    def set_grid(self, grid):
        self._R      = grid.Rc
        self._set_arrays()

    def _set_arrays(self):
        R  = self._R
        self._cs     = self._f_cs(R)
        self._H      = self._f_H(R)
        self._nu     = self._f_nu(R)
        self._alpha  = self._f_alpha(R)

    @property
    def cs(self):
        return self._cs

    @property
    def H(self):
        return self._H

    @property
    def nu(self):
        return self._nu

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    @property
    def mu(self):
        return self._mu

    def update(self, dt, Sigma, amax=None, star=None):
        """Update the eos"""
        pass

    def ASCII_header(self):
        """Print eos header"""
        head = '# {} gamma: {}, mu: {}'
        return head.format(self.__class__.__name__,
                           self.gamma, self.mu)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        def fmt(x):  return "{}".format(x)
        return self.__class__.__name__, { "gamma" : fmt(self.gamma),
                                          "mu" : fmt(self.mu) }
    
class LocallyIsothermalEOS(EOS_Table):
    """Simple locally isothermal power law equation of state:

    args:
        h0      : aspect ratio at 1AU
        q       : power-law index of sound-speed
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.4
    """
    def __init__(self, star, h0, q, alpha_t, mu=2.4):
        super(LocallyIsothermalEOS, self).__init__()
        
        self._h0 = h0
        self._cs0 = h0 * star.M**0.5
        self._q = q
        self._alpha_t = alpha_t
        self._H0 = h0
        self._T0 = (AU*Omega0)**2 * mu / GasConst
        self._mu = mu
        
    def _f_cs(self, R):
        return self._cs0 * R**self._q

    def _f_H(self, R):
        return self._H0 * R**(1.5+self._q)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        return self._alpha_t

    @property
    def T(self):
        return self._T0 * self.cs**2

    @property
    def Pr(self):
        return np.zeros_like(self._R)

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(LocallyIsothermalEOS, self).ASCII_header()
        head += ', h0: {}, q: {}, alpha: {}'
        return head.format(self._h0, self._q, self._alpha_t)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(LocallyIsothermalEOS, self).HDF5_attributes()
        head["h0"]   = "{}".format(self._h0)
        head["q"]     = "{}".format(self._q)
        head["alpha"] = "{}".format(self._alpha_t)
        return name, head

    @staticmethod
    def from_file(filename):
        raise NotImplementedError('')

    @property
    def star(self):
        return self._star


class SimpleDiscEOS(EOS_Table):
    """Simple approximate irradiated/viscous equation of state from Liu et al.
    (2019).

    args:
        alpha_t : turbulent alpha parameter
        star    : stellar properties
        mu      : mean molecular weight, default=2.33
        K0      : Opacity constant (K = K0 T), default = 0.01
    """
    def __init__(self, star, alpha_t, mu=2.33, K0=0.01):
        super(SimpleDiscEOS, self).__init__()
        

        self._alpha_t = alpha_t
        self._mu = mu
        self._K0 = K0
        self._star = star
        
        self._Tnu = np.sqrt(27/64*alpha_t*Omega0*GasConst*K0/(mu*sig_SB))

        self._set_constants()
        
    # def _f_cs(self, R):
    #     return self._cs0 * R**self._q

    # def _f_H(self, R):
    #     return self._H0 * R**(1.5+self._q)
    
    # def _f_nu(self, R):
    #     return self._alpha_t * self._f_cs(R) * self._f_H(R)

    # def _f_alpha(self, R):
    #     return self._alpha_t

    def _set_constants(self):
        star = self._star

        Ls = star.Rs**2 * (star.T_eff / 5770)**4
        self._Tirr0 = 150 * Ls**(2/7.) * star.M**(-4/7)
        self._Tnu0 = self._Tnu * star.M**0.25

        self._cs0 = (Omega0**-1/AU) * (GasConst / self._mu)**0.5
        self._H0  = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5

    def update(self, dt, Sigma, amax=1e-5, star=None):
        if star:
            self._star = star

        self._set_constants()

        Tirr = self._Tirr0 * self._R**(-3/7.)
        Tvis = self._Tnu * Sigma * self._R**-0.75

        self._T = (Tirr**4 + Tvis**4)**0.25
        self._Sigma = Sigma
        
        self._set_arrays()

    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self):
        super(SimpleDiscEOS,self)._set_arrays()
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._cs0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        return self._alpha_t

    def _f_Pr(self):
        kappa = self._K0 * self._T
        tau = 0.5 * self._Sigma * kappa
        f_esc = 1 + 2/(3*tau*tau)
        Pr_1 =  2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr

    def ASCII_header(self):
        """LocallyIsothermalEOS header string"""
        head = super(SimpleDiscEOS, self).ASCII_header()
        head += ', alpha: {}, mu: {}, K0= {}'
        return head.format(self._alpha_t, self._mu, self._K0)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(SimpleDiscEOS, self).HDF5_attributes()
        head["alpha"] = "{}".format(self._alpha_t)
        head['mu'] = "{}".format(self._mu)
        head['K0'] = "{}".format(self._K0)
        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "SimpleDiscEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if key == 'mu' or key == 'K0':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())


        return SimpleDiscEOS(star, alpha, **kwargs)


    @property
    def star(self):
        return self._star


_sqrt2pi = np.sqrt(2*np.pi)
class IrradiatedEOS(EOS_Table):
    """Model for an active irradiated disc.

    From Nakamoto & Nakagawa (1994), Hueso & Guillot (2005).

    args:
        star    : Stellar properties
        alpha_t : Viscous alpha parameter
        Tc      : External irradiation temperature (nebular), default=10
        Tmax    : Maximum temperature allowed in the disc, default=1500
        mu      : Mean molecular weight, default = 2.4
        gamma   : Ratio of specific heats
        kappa   : Opacity, default=Zhu2012
        accrete : Whether to include heating due to accretion,
                  default=True
    """
    def __init__(self, star, alpha_t, Tc=10, Tmax=1500., mu=2.4, gamma=1.4,
                 kappa=None,
                 accrete=True, tol=None): # tol is no longer used
        super(IrradiatedEOS, self).__init__()

        self._star = star
        
        self._dlogHdlogRm1 = 2/7.

        self._alpha_t = alpha_t
        
        self._Tc = Tc
        self._Tmax = Tmax
        self._mu = mu

        self._accrete = accrete
        self._gamma = gamma

        if kappa is None:
            self._kappa = opacity.Zhu2012
        else:
            self._kappa = kappa
        
        self._T = None

        self._compute_constants()

    def _compute_constants(self):
        self._sigTc4 = sig_SB*self._Tc**4
        self._cs0 = (Omega0**-1/AU) * (GasConst / self._mu)**0.5
        self._H0  = (Omega0**-1/AU) * (GasConst / (self._mu*self._star.M))**0.5


    def update(self, dt, Sigma, amax=1e-5, star=None):
        if star:
            self._star = star
            self._compute_constants()
        star = self._star
            
        # Temperature/gensity independent quantities:
        R = self._R
        Om_k = Omega0 * star.Omega_k(R)

        X = star.Rau/R
        f_flat  = (2/(3*np.pi)) * X**3
        f_flare = 0.5 * self._dlogHdlogRm1 * X**2
        
        # Heat capacity
        mu = self._mu
        #C_V = (k_B / (self._gamma - 1)) * (1 / (mu * m_H))
        
        alpha = self._alpha_t
        if not self._accrete:
            alpha = 0.

        # Local references 
        max_heat = sig_SB * (self._Tmax*self._Tmax)*(self._Tmax*self._Tmax)
        star_heat = sig_SB * star.T_eff**4
        sqrt2pi = np.sqrt(2*np.pi)            
        def balance(Tm):
            """Thermal balance"""
            cs = np.sqrt(GasConst * Tm / mu)
            H = cs / Om_k

            kappa = self._kappa(Sigma / (sqrt2pi * H), Tm, amax)
            tau = 0.5 * Sigma * kappa
            H /= AU

            # External irradiation
            dEdt = self._sigTc4
            
            # Compute the heating from stellar irradiation
            dEdt += star_heat * (f_flat + f_flare * (H/R))

            # Viscous Heating
            visc_heat = 1.125*alpha*cs*cs * Om_k
            dEdt += visc_heat*(0.375*tau*Sigma + 1./kappa)
            
            # Prevent heating above the temperature cap:
            dEdt = np.minimum(dEdt, max_heat)

            # Cooling
            Tm2 = Tm*Tm
            dEdt -= sig_SB * Tm2*Tm2

            # Change in temperature
            return (dEdt/Omega0) # / (C_V*Sigma)

        # Solve the balance using brent's method (needs ~ 20 iterations)
        T0 = self._Tc
        T1 = self._Tmax
        if self._T is not None:
            dedt = balance(self._T)
            T0 = np.where(dedt > 0, self._T, T0)
            T1 = np.where(dedt < 0, self._T, T1)

        self._T =  brentq(balance, T0, T1)
        self._Sigma = Sigma

        # Save the opacity:
        cs = np.sqrt(GasConst * self._T / mu)
        H = cs / Om_k
        self._kappa_arr = self._kappa(Sigma / (sqrt2pi * H), self._T, amax)
        
        self._set_arrays()


    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = None

    def _set_arrays(self):
        super(IrradiatedEOS,self)._set_arrays()
        self._Pr = self._f_Pr()
    
    def __H(self, R, T):
        return self._H0 * np.sqrt(T * R*R*R)

    def _f_cs(self, R):
        return self._cs0 * self._T**0.5

    def _f_H(self, R):
        return self.__H(R, self._T)
    
    def _f_nu(self, R):
        return self._alpha_t * self._f_cs(R) * self._f_H(R)

    def _f_alpha(self, R):
        return self._alpha_t

    def _f_Pr(self):
        kappa = self._kappa_arr
        tau = 0.5 * self._Sigma * kappa
        f_esc = 1 + 2/(3*tau*tau)
        Pr_1 =  2.25 * self._gamma * (self._gamma - 1) * f_esc
        return 1. / Pr_1

    @property
    def T(self):
        return self._T

    @property
    def Pr(self):
        return self._Pr

    @property
    def star(self):
        return self._star

    def ASCII_header(self):
        """IrradiatedEOS header"""
        head = super(IrradiatedEOS, self).ASCII_header()
        head += ', opacity: {}, T_extern: {}K, accrete: {}, alpha: {}'
        head += ', Tmax: {}K'
        return head.format(self._kappa.__class__.__name__,
                           self._Tc, self._accrete, self._alpha_t,
                           self._Tmax)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        name, head = super(IrradiatedEOS, self).HDF5_attributes()

        head["opacity"]  = self._kappa.__class__.__name__
        head["T_extern"] = "{} K".format(self._Tc)
        head["accrete"]  = "{}".format(bool(self._accrete))
        head["alpha"]    = "{}".format(self._alpha_t)
        head["Tmax"]     = "{} K".format(self._Tmax)

        return name, head

    @staticmethod
    def from_file(filename):
        import star

        star = star.from_file(filename)
        alpha = None

        with open(filename) as f:
            for line in f:
                if not line.startswith('#'):
                    raise AttributeError("Error: EOS type not found in header")
                elif "IrradiatedEOS" in line:
                    string = line 
                    break 
                else:
                    continue

        kwargs = {}
        for item in string.split(','):    
            key, val = [ x.strip() for x in item.split(':')]

            if   key == 'gamma' or key == 'mu':
                kwargs[key] = float(val.strip())
            elif key == 'alpha':
                alpha = float(val.strip())
            elif key == 'accrete':
                kwargs[key] = bool(val.strip())
            elif key == 'T_extern':
                kwargs['Tc'] = float(val.replace('K','').strip())

        return IrradiatedEOS(star, alpha, **kwargs)

def from_file(filename):
    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                raise AttributeError("Error: EOS type not found in header")
            elif "IrradiatedEOS" in line:
                return IrradiatedEOS.from_file(filename)      
            elif "SimpleDiscEOS" in line:
                return SimpleDiscEOS.from_file(filename)
            elif "LocallyIsothermalEOS" in line:
                return LocallyIsothermalEOS.from_file(filename)
            else:
                continue


import astropy.units as u

boltz = 1.3806e-16 # boltzmann constant
mH = 1.67e-24 # mass of hydrogen atom in grams
mu = 2.34 
gamma = 1.4
sig_SB = 5.6704e-5# Stefan-Boltzmann constant, cgs
G = 6.67430e-8 # Gravitational constant, cgs
Msun = 1.989e33     # Solar mass in grams
AU = 1.496e13       # 1 AU in cm
yr = 3.165e7        # 1 yr in s

class ChambersEOS(EOS_Table):
    '''
    EOS only to be used with the Chambers DiscWindEvolution model. This model is self-contained and calculates
    its own temperature. Use the update_params() method to sync init params with DiscWindEvolution.
    '''
    
    def __init__(self, star, sigma0, r0, T0, v0, fw, K, Tevap, rexp, k0): # tol is no longer used
        super(ChambersEOS, self).__init__()

        self._star = star
        self.sigma0 = sigma0
        self.r0 = r0*AU
        self.T0 = T0
        self.v0 = v0
        self.fw = fw
        self.K = K
        self.Tevap = Tevap
        self.rexp = rexp*AU
        self.k0 = k0
        
        self._tol=0.5

        self._eos_time = 0

        self.config = Config

        self.cs0 = ((Constants.gamma)*(Constants.boltz)*self.T0/(Constants.mu*Constants.mH))**0.5

    def set_grid(self, grid):
        self._R = grid.Rc
        self._T = self.calculate_T(0, self._star)

    def calculate_T(self, t, star):
        
        # dt = ((self._time-dt)/(2*np.pi))
            
        # mask = R > self.edge
        # try:
        #     self.fw = np.where(mask, self.deadzone_fw, self.fw)
        #     self.v0 = np.where(mask, self.deadzone_v0, self.v0)
        # except AttributeError:
        #     pass

        R=self._R*AU
        t*=yr

        Mstar = star.M * Msun  # Convert star mass to grams
        s0 = np.sqrt(self.rexp / self.r0)
        V = self.Tevap / self.T0

        # Calculate constant A in CGS units
        Atop = 9 * (1 - self.fw) * G * Mstar * self.k0 * (self.sigma0 ** 2) * self.v0
        Abottom = 32 * sig_SB * (self.r0 ** 2) * (self.T0 ** 4)
        A = np.sqrt(Atop / Abottom)

        x = (R / self.r0) ** 0.5

        # Equation 39 terms
        p0 = A * (V ** 0.5) * ((A ** 2 + 1) / (A ** 2 + V ** 3)) ** (1 / 6)
        J = self.fw / (1 - self.fw)
        common_term = ((1 + J) ** 2 + 8 * J * self.K) ** 0.5
        n = -1 - (2 / 5) * common_term
        b = ((1 - J) / 2) + (1 / 2) * common_term
        tau = (8 * self.r0 * s0 ** (5 / 2)) / (25 * self.v0 * (1 - self.fw))

        # Equation 38 terms
        time_factor = (1 + (t / tau)) ** n
        p1 = p0 * time_factor * (x ** b)
        p2 = np.exp((1 / s0) ** (5 / 2) - (x / s0) ** (5 / 2) * (1 + t / tau) ** -1)
        p = p1 * p2

        # Surface density (sigma) calculation
        sigma_1 = (self.sigma0 / A) * p * x ** (-5 / 2)
        sigma_2 = (1 + V ** (-2) * p * (x ** (-9 / 2))) / (1 + p * (x ** (-5 / 2)))
        sigma = sigma_1 * sigma_2 ** 0.25

        # Temperature (T) calculation
        sig = A * sigma / self.sigma0
        Ttop = sig ** 2 + 1
        Tbottom = sig ** 2 + V ** 3 * x ** 3
        T = self.Tevap * (Ttop / Tbottom) ** (1 / 3)
        self._T = T

        return T

    def update(self, dt, Sigma, amax=1e-5, star=None):
        if star == None:
            star = self._star
        self._eos_time += dt/(2*np.pi)
        t = self._eos_time
        self._T = self.calculate_T(t, star)

        self._set_arrays()

    def _Omega(self, R):
        Omega = self._star.Omega_k(R)
        return Omega*(2*np.pi)/yr

    def _f_cs(self, R=None):
        k_B = Constants.boltz
        m_H = Constants.mH
        T = self._T
        cs = (k_B*T/(Constants.mu*m_H))**0.5
        cs = cs*2.10805e-6/(2*np.pi) # convert to au/yr
        # cs = np.sqrt(GasConst * self._T / mu) * (Omega0**-1/AU)
        return cs

    def _f_alpha(self, R=None):
        alpha_turb = ((1 - self.fw)*self.r0*self.v0*self.Omega_ref / (self.cs0**2))
        alpha_wind = (self.fw*self.v0 / (self.cs0))
        self._alpha_t = alpha_turb+alpha_wind
        return self._alpha_t
    
    def _f_H(self, R):
        Omega = self._star.Omega_k(R)
        _H = self._f_cs(R)/Omega
        return _H
    
    def _f_nu(self, R):
        alpha = self._f_alpha(R)
        H = self._f_H(R)
        Omega = self._star.Omega_k(R)
        nu = alpha*H**2*Omega
        return nu
    
    def update_params(self, disk_wind):
        """Sync parameters from DiskWindEvolution"""
        self.sigma0 = disk_wind.sigma0
        self.r0 = disk_wind.r0
        self.T0 = disk_wind.T0
        self.v0 = disk_wind.v0
        self.fw = disk_wind.fw
        self.K = disk_wind.K
        self.Tevap = disk_wind.Tevap
        self.rexp = disk_wind.rexp
        self.k0 = disk_wind.k0
    
    @property
    def Omega_ref(self):
        return self._star.Omega_k(self.r0/AU)*(2*np.pi)/yr
    
    @property
    def Pr(self):
        return np.zeros_like(self._R)

    @property
    def T(self):
        return self._T

    @property
    def star(self):
        return self._star
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from DiscEvolution.star import SimpleStar
    from DiscEvolution.grid import Grid

    alpha = 1e-3
    star = SimpleStar(M=1.0, R=3.0, T_eff=4280.)

    active  = IrradiatedEOS(star, alpha)
    passive = IrradiatedEOS(star, alpha, accrete=False)
    marco   = IrradiatedEOS(star, alpha, kappa=opacity.Tazzari2016())

    powerlaw = LocallyIsothermalEOS(star, 1/30., -0.25, alpha)

    grid = Grid(0.1, 500, 1000, spacing='log')
    
    Sigma = 2.2e3 / grid.Rc**1.5

    amax = 10 / grid.Rc**1.5
    
    # c  = { 'active' : 'r', 'passive' : 'b', 'marco' : 'm',
    #        'chambers' : 'g' }
    # ls = { 0 : '-', 1 : '--' }
    # for i in range(2):
    #     for eos, name in [[active, 'active'],
    #                       [marco, 'marco'],
    #                       [passive, 'passive'],
    #                       [chambers, 'chambers']]:
    #         eos.set_grid(grid)
    #         eos.update(0, Sigma, amax=amax)

    #         label = None
    #         if ls[i] == '-':
    #             label = name
                
    #         plt.loglog(grid.Rc, eos.T, c[name] + ls[i], label=label)
    #     Sigma /= 10
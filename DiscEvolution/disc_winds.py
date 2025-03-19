import astropy.units as u
import numpy as np
from .chambers_config import Constants
import numpy as np

# Constants in CGS units (without Astropy)
sig_SB = 5.6704e-5  # Stefan-Boltzmann constant in erg / cm^2 / s / K^4
G = 6.67430e-8      # Gravitational constant in cm^3 / g / s^2
Msun = 1.989e33     # Solar mass in grams
AU = 1.496e13       # 1 AU in cm
yr = 3.165e7        # 1 yr in s

class DiskWindEvolution:
    '''This class contains the equations for the disk wind evolution.
    Mainly stuff from Chambers et al. 2019, supplemented with Alessi et al. 2022.'''

    def __init__(self, star, sigma0, r0, T0, v0, fw, K, Tevap, rexp, k0, edge=0):

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
        self.edge = edge

        self._tol = 10

        self.cs0 = ((Constants.gamma)*(Constants.boltz)*self.T0/(Constants.mu*Constants.mH))**0.5

        self._time = 0

    def __call__(self, dt, disc, tracers=None, adv=None):
        '''Returns the surface density, temperature, total mass, and accretion rate at a given time t and radius R.'''
        star = self._star

        self._time += dt/(2*np.pi)
        t = self._time
        # mask = R > self.edge
        # try:
        #     self.fw = np.where(mask, self.deadzone_fw, self.fw)
        #     self.v0 = np.where(mask, self.deadzone_v0, self.v0)
        # except AttributeError:
        #     pass
        type = "disc"
        try:
            R = disc.R
        except: 
            R = disc
            type = "grid"
        
        R=R*AU
        t=t*yr
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

        # Calculate total mass and accretion rate
        dA = np.pi * (R[1:] ** 2 - R[:-1] ** 2)
        dM = sigma[1:] * dA
        Mtot = np.sum(dM)  # in grams

        v_in = self.v0 * (T[0] / self.T0) ** 0.5  # inward velocity
        Macc = 2 * np.pi * R[0] * sigma[0] * v_in  # accretion rate in g/s

        # Convert total mass and accretion rate to Solar units
        Mtot /= Msun  # Total mass in solar masses
        Macc /= Msun / 3.154e7  # Accretion rate in solar masses per year

        self._set_constants(T, Mtot, Macc)

        if type == "disc":
            disc.Sigma[:] = sigma
        else:
            return sigma
    
    def get_suzuki_params(self):
        """
        Convert Chambers parameters (fw, v0) to alpha_turb and alpha_wind
        """
        alpha_turb = ((1 - self.fw)*self.r0*self.v0*self.Omega_ref / (self.cs0**2))
        alpha_wind = (self.fw*self.v0 / (self.cs0))

        return alpha_turb, alpha_wind
    
    def calculate_chambers_params(self, alpha_turb, alpha_wind, set = False):
        """
        Calculates Chamber's params (fw, v0) from Suzuki (alpha_turb, alpha_wind)
        set: Automatically set the calculated params outside the dead zone
        """

        A = self.r0*self.Omega_ref/self.cs0**2
        B = 1/self.cs0
        v0 = (alpha_turb+A/B*alpha_wind)/A
        fw = alpha_wind*self.cs0/v0

        if set:
            self.fw = fw
            self.v0 = v0
        else:
            return fw, v0
        
    def viscous_velocity(self, disc, Sigma=None):
        Sigma = disc.Sigma
        R = disc.R
        nu = disc.nu

        # v_r = -3/(Sigma*np.sqrt(R)) * np.diff(nu*Sigma*np.sqrt(R)) / np.diff(R)
        v_r = -3 / (Sigma[:-1] * np.sqrt(R[:-1])) * np.diff(nu * Sigma * np.sqrt(R)) / np.diff(R)
    
        return v_r
    
    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc
        
    def set_dead_zone_params(self, alpha_turb, alpha_wind):
        self.deadzone_fw, self.deadzone_v0 = self.calculate_chambers_params(alpha_turb, alpha_wind)

    def _set_constants(self, T, M_tot, M_acc):
        self._T = T
        self._M_tot = M_tot
        self._M_acc = M_acc


    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}
    
    @property
    def Omega_ref(self):
        return self._star.Omega_k(self.r0/AU)*(2*np.pi)/yr
    
    @property
    def M_acc(self):
        """Accretion rate onto star"""
        return self._M_acc
    
    @property
    def M_tot(self):
        """Total disk mass"""
        return self._M_tot
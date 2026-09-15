from __future__ import print_function
import numpy as np
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline as ispline
from scipy.interpolate import UnivariateSpline as spline
from scipy.integrate import ode
from DiscEvolution.constants import *
from DiscEvolution.disc_utils import make_ASCII_header
from DiscEvolution.grid import reduce

################################################################################
# Planet collections class
################################################################################

class Planets(object):
    """
    Data for growing planets.

    Holds the location, core & envelope mass, and composition of growing
    planets.

    args:
        Nchem    : number of chemical species to track, default = None
    """

    def __init__(self, Nchem=None):
        self.R  = np.array([], dtype='f4')
        self.M_core = np.array([], dtype='f4')
        self.M_env  = np.array([], dtype='f4')
        self.t_form = np.array([], dtype='f4')
        self.Mdot = np.array([], dtype='f4')
        self._R_capt  = np.array([], dtype='f4')

        self._N = 0

        if Nchem:
            self.X_core = np.array([[] for _ in range(Nchem)], dtype='f4')
            self.X_env  = np.array([[] for _ in range(Nchem)], dtype='f4')

        else:
            self.X_core = None
            self.X_env  = None
        self._Nchem = Nchem

    def add_planet(self, t, R, Mcore, Menv, X_core=None, X_env=None):
        """Add a new planet"""
        if self._Nchem:
            self.X_core = np.c_[self.X_core, X_core]
            self.X_env  = np.c_[self.X_env, X_env]

        self.R      = np.append(self.R, R)
        self.M_core = np.append(self.M_core, Mcore)
        self.M_env  = np.append(self.M_env, Menv)
        self._R_capt  = np.append(self._R_capt, 0)
        self.Mdot = np.append(self.Mdot,0)
        self.t_form = np.append(self.t_form, np.ones_like(Menv)*t)

        self._N += 1

    def append(self, planets):
        """Add a list of planets from another planet object"""
        self.add_planet(planets.t_form, planets.R,
                        planets.M_core, planets.M_env,
                        planets.X_core, planets.X_env)

    @property
    def M(self):
        return self.M_core + self.M_env

    @property
    def N(self):
        """Number of planets"""
        return self._N

    @property
    def chem(self):
        if self._Nchem is None:
            return False
        return self._Nchem > 0
    
    @property
    def R_capt(self):
        """Capture radius of the planet"""
        return self._R_capt

    def __getitem__(self, idx):
        """Get a sub-set of the planets"""
        sub = Planets(self._Nchem)

        sub.R      = self.R[idx]
        sub.M_core = self.M_core[idx]
        sub.M_env  = self.M_env[idx]
        sub.t_form = self.t_form[idx]
        if self.chem:
            sub.X_core = self.X_core[...,idx]
            sub.X_env  = self.X_env[...,idx]

        try:
            sub._N = len(sub.R)
        except TypeError:
            sub._N = 1

        return sub

    def __iter__(self):
        for i in range(self.N):
            yield self[i]
    
################################################################################
# Accretion
################################################################################

class GasAccretion(object):
    """
    Gas giant accretion model of Bitsch et al (2015).

    Combines models from Piso & Youdin (2014) for accretion onto low mass
    envelopes and Machida et al (2010) for accretion onto massive envelopes.

    args:
        General:
           disc  : Accretion disc
           f_max : maximum accretion rate relative to disc accretion rate,
                   default=0.8

        Piso & Youdin parameters:
           f_py      : accretion rate fitting factor, default=0.2
           kappa_env : envelope opacity [cm^2/g], default=0.06
           rho_core : core density [g cm^-3], default=5.5
    """

    def __init__(self, disc, f_max=0.8, f_py=0.2, kappa_env=0.05, rho_core=5.5):

        # General properties
        self._fmax = f_max # depreciated with the addition of winds.  MLB - restored.
        self._disc = disc

        # Piso & Youdin parameters
        self._fPiso = 0.1 * 1.75e-3 / f_py**2
        self._fPiso /= kappa_env * (rho_core/5.5)**(1/6.)
        # Convert Mearth / M_yr to M_E Omega0**-1
        self._fPiso *= 1e-6 / (2*np.pi)

        head = {"f_max"     : "{}".format(f_max),
                "f_py"      : "{}".format(f_py),
                "kappa_env" : "{} cm^2 g^-1".format(kappa_env),
                "rho_core"  : "{} g cm^-1".format(rho_core),
                }
        
        self._head = (self.__class__.__name__, head)

    def ASCII_header(self):
        """Get header details"""
        return make_ASCII_header(self.HDF5_attributes())

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self._head

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def computeMdot(self, Rp, M_core, M_env):
        """
        Compute gas accretion rate.

        args:
            Rp     : radius, AU
            M_core : Core mass, Mearth
            M_env  : Envelope mass, Mearth

        returns:
            Mdot : accretion rate in Mearth per Omega0**-1
        """

        # Cache data:
        Mp = M_core + M_env

        disc = self._disc
        
        # Piso & Youdin (2014) accretion rate:
        T81 = self._disc.interp(Rp, disc.T)/81
        # MLB fudge to stop failure when M_env = 0
        M_env = np.where(M_env == 0, 1.e-4*M_core, M_env)
        Mdot_PY = self._fPiso * T81**-0.5 * M_core**(11/3.) / M_env
        
        # Machida+ (2010) accretion rate
        star = self._disc.star
        rH = star.r_Hill(Rp, Mp*Mearth/Msun)

        Sig = disc.interp(Rp, disc.Sigma_G)
        H   = disc.interp(Rp, disc.H)
        nu   = disc.interp(Rp, disc.nu)
        #nu  = self._disc.interp(Rp, self._disc.nu)

        Om_k = star.Omega_k(Rp)
        
        # Accretion rate is the minimum of two branches, meeting at
        # rH/H ~ 0.3
        f = np.minimum(0.83 * (rH/H)**4.5, 0.14)
        
        # Convert to Mearth / AU**2
        Sig /= Mearth/AU**2

        Mdot_Machida = f * Om_k * Sig * H*H

        Mdot = np.where(M_core > M_env, Mdot_PY, Mdot_Machida)
        
        disc = self._disc

        # generalized limit for winds and viscous case (added by Yuvan S., 2025.)
        #Mdot_limit = 2*np.pi * Rp * Sig * np.abs(np.interp(Rp, disc._grid.Re[1:-1], disc._gas.viscous_velocity(disc)))
        #  This is not correct - limit is supposed to be fmax of the accretion rate 
        Mdot_limit = self._fmax * 2*np.pi * Rp * Sig * np.abs(np.interp(Rp, disc._grid.Re[1:-1], disc._gas.viscous_velocity(disc)))
        # Original:
        #Mdot_limit = self._fmax * 3*np.pi*Sig*nu
        return np.minimum(Mdot, Mdot_limit)

    def __call__(self, planets):
        """
        Compute gas accretion onto planets

        args:
             planets : planets object.

        returns:
            Mdot : accretion rate in Mearth per Omega0**-1
        """
        return self.computeMdot(planets.R, planets.M_core, planets.M_env)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        pass
    
class PebbleAccretion(object):
    """
    Pebble accretion model of Bitsch+ (2015) with Bondi regime added.

    See also, Lambrechts & Johansen (2012) for Bondi regime, Morbidelli+ (2015) for Hill regime.
    """

    def __init__(self, disc):
        self.set_disc(disc)

    def ASCII_header(self):
        """Get header details"""
        return '# {}'.format(self.__class__.__name__)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, {}

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def M_iso(self, R):
        """Pebble isolation mass."""
        h = self._disc.interp(R, self._disc.H) / R
        return 20. * (h/0.05)**3

    def M_transition(self, R, epsilon=None):
        """
        Compute the transition mass between Bondi and Hill Regimes.

        args:
            R : radius, AU
            epsilon : approximate power law scaling of pressure with radius

        returns:
            M_t (ndarray): transition mass, Mearth
        """

        h = self._disc.interp(R, self._disc.H) / R
        
        if not epsilon is None:
            eta = 0.5 * h**2 * epsilon
        else:
            # Use a safe, noise free approximation here
            eta = - 0.5 * h*h * (-2.75)

        Om_k = self._disc.star.Omega_k(R)
        v_k = Om_k * R
        
        M_t = (1/3.)**0.5 * (eta*v_k)**3 / (G * Om_k) * Msun / Mearth
        return M_t
    
    def Mdot_Hill(self, Rp, Mp):
        """
        Compute the pebble accretion rate in the Hill regime, according to  Morbidelli+ (2015).
        
        args:
            Rp : heliocentric radius of planet, AU
            Mp : mass of planet, M_earth

        returns:
            Mdot (ndarray): Mass accretion rate of pebbles in Hill regime for each planet.
        """

        # Cache local varibales
        disc = self._disc
        star = disc.star
        
        # Interpolate disc properites to planet location
        Hp    = disc.interp(Rp, disc.Hp[1])
        St    = disc.interp(Rp, disc.Stokes()[1])
        Sig_p = disc.interp(Rp, disc.Sigma_D[1])

        # Radius at which gravity of star takes over gravity of planet
        rH   = star.r_Hill(Rp, Mp*Mearth/Msun) 
        r_eff = rH * (St/0.1)**(1/3.)

        Sig_p /= Mearth / AU**2
        
        # Accretion rate in the limit Hp << rH
        Mdot = 2*np.minimum(rH*rH, r_eff*r_eff) * star.Omega_k(Rp) * Sig_p

        # 3D correction for Hp >~ r_H:
        # Replaces Sigma_p -> np.pi * rho_pltsml * r_eff
        Mdot *= np.minimum(1, r_eff *(np.pi/8)**0.5 / Hp)

        return Mdot
    
    def Mdot_Bondi(self, Rp, Mp, epsilon):
        """
        Compute the pebble accretion rate in the Bondi regime, according to Lambretchs and Johansen (2012).
        
        args:
            Rp : heliocentric radius of planet, AU
            Mp : mass of planet, M_earth
            epsilon : approximate power law scaling of pressure with radius

        returns:
            Mdot (ndarray): Mass accretion rate of pebbles in Bondi regime for each planet.
        """

        # Cache local varibales
        disc = self._disc
        star = disc.star

        # Interpolate disc properites to planet location
        Hp    = disc.interp(Rp, disc.Hp[1])
        St    = disc.interp(Rp, disc.Stokes()[1])
        Sig_p = disc.interp(Rp, disc.Sigma_D[1])*(AU**2 / Mearth)

        # approximate relative velocity between pebbles and planet
        delta_v = epsilon * star.Omega_k(Rp) * Rp 

        r_B = G * Mp * (Mearth/Msun) / delta_v**2  

        # Find effective accretion radius
        tf = St/star.Omega_k(Rp)
        tB = r_B/delta_v
        r_d = r_B * (tB/tf)**(-0.5)

        rho_peb = Sig_p / (Hp * np.sqrt(2 * np.pi)) 

        # Find 3D to 2D transition mass for Bondi regime
        M_3D_to_2D = Hp * delta_v**2 * (tB/tf)**(0.5) / (G * Mearth/Msun)

        # compute mass accretion rate based on 2D or 3D regime
        Mdot = np.where(Mp < M_3D_to_2D,
            np.pi * rho_peb * r_d**2 * delta_v,
            2 * r_d * Sig_p * delta_v)

        return np.array(Mdot)

    def computeMdot(self, Rp, Mp):
        """
        Calculate the pebble accretion rate.
    
        args:
             Rp : radius of planet in AU
             Mp : mass of planet in M_earth

        returns:
            Mdot (ndarray): Mass accretion rate of pebbles for each planet.
        """

        disc = self._disc

        # Interpolate disc properites to planet location
        St    = disc.interp(Rp, disc.Stokes()[1])
        epsilon = np.abs((np.diff(np.log(disc.P))) / (np.diff(np.log(disc.grid.Rc))))
        epsilon = np.insert(epsilon, 0, epsilon[0])  # Epsilon is approximately constant at small radii.
        epsilon = disc.interp(Rp, epsilon)

        M_transition = self.M_transition(Rp, epsilon)

        # compute mass accretion rate based on Bondi or Hill regime
        Mdot = np.where(Mp < (M_transition/(8 * St)), self.Mdot_Bondi(Rp, Mp, epsilon), self.Mdot_Hill(Rp, Mp))

        # Mdot=0 if planet mass is above pebble isolation mass
        return np.array(Mdot) * (Mp < self.M_iso(Rp))

    def __call__(self, planets):
        """Compute pebble accretion rate"""
        return self.computeMdot(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        
        lgP = spline(np.log(self._disc.R), np.log(self._disc.P))
        self._dlgP = lgP.derivative(1)



class PlanetesimalAccretion(object):
    """
    Planetesimal accretion model.

    args:
        disc: disc object
        Mdot_migrate: function to compute accretion rate during migration
        Mdot_insitu: function to compute accretion rate in-situ
        rho_core: core density [g cm^-3], default = 5.5
        gamma: stirring parameter, default = None
    """

    def __init__(self, disc, Mdot_migrate = True, Mdot_insitu = True, rho_core = 5.5, gamma = None):

        if gamma is None:
            self._stirring = np.sqrt(disc.alpha)*disc.h
        else:
            self._stirring = gamma*np.ones_like(disc.R)

        self.set_disc(disc)

        self.rho_core = rho_core

        self.dRdt = None
        self._Mdot_migrate = Mdot_migrate
        self._Mdot_insitu = Mdot_insitu

    def set_disc(self, disc):
        self._disc = disc

    # Fortier et al 2013 accretion model
    
    def M_iso_pltsml(self, Rp):
        """
        Compute the planetesimal isolation mass (Rafikov 2011).

        Rp: Protoplanet location (in AU)

        return: Planetesimal isolation mass (in Earth masses)
        """

        disc = self._disc

        b_tilde = 10 # dimensionless spacing parameter
        Mstar = disc.star.M * Msun
        Sigma_D = disc.interp(Rp, disc.Sigma_D[2])

        return (2 * np.pi * (Rp * AU) ** 2 * b_tilde * Sigma_D) ** (3/2) * (3 * Mstar) ** (-1/2) / Mearth
    
    def R_core(self, Mp):
        """
        Compute the physical radius of the protoplanet core assuming a constant density (same as planetesimal density).

        Mp: Protoplanet mass (in Earth masses)

        return: Protoplanet core physical radius (in AU)
        """

        rho_core = self.rho_core

        return (3 * Mp * Mearth / (4 * np.pi * rho_core)) ** (1/3) / AU
    
    def _I_F(self, beta):
        """
        Computes the numerical elliptic integral approximation (Fortier et al 2012).

        beta: Inclination to eccentricity ratio
        
        return: Integral approximation
        """
        return (1 + 0.95925 * beta + 0.77251 * beta ** 2) / (beta * (0.13142 + 0.12295 * beta))

    def _I_G(self, beta):
        """
        Computes the numerical elliptic integral approximation (Fortier et al 2012).

        beta: Inclination to eccentricity ratio
        
        return: Integral approximation
        """
        return (1 + 0.3996 * beta) / (beta * (0.0369 + 0.048333 * beta + 0.006874 * beta ** 2))
    
    def P_coll(self, e2, i2, Rp, Mp):
        """
        Compute the probability that the planetesimal is accreted by the embryo (Fortier et al 2012).
        
        e2: planetesimal eccentricity squared
        i2: planetesimal inclination squared
        Rp: Protoplanet location (in AU)
        Mp: Protoplanet mass (in Earth masses)

        return: Probability that the planetesimal is accerted by the embryo
        """

        disc = self._disc

        Mstar = disc.star.M * Msun
        rH = disc.star.r_Hill(Rp, Mp * Mearth / Msun) * AU
        Rpltsml = disc.interp(Rp, disc.R_planetesimal) * AU
        Rcore = self.R_capt(Rp, Mp) * AU

        e_tilde = np.sqrt(e2) / (Mp * Mearth / (3 * Mstar)) ** (1/3)
        i_tilde = np.sqrt(i2) / (Mp * Mearth / (3 * Mstar)) ** (1/3)
        beta = i_tilde / e_tilde

        P_low = 11.3 * ((Rcore + Rpltsml) / rH) ** (1/2)
        P_med = ((Rcore + Rpltsml) ** 2 / (4 * np.pi * rH ** 2) * i_tilde) * (17.3 + 232 * rH / (Rcore + Rpltsml))
        P_high = ((Rcore + Rpltsml) ** 2 / (2 * np.pi * rH ** 2)) * (self._I_F(beta) + 6 * rH * self._I_G(beta) / ((Rcore + Rpltsml) * e_tilde ** 2))

        return np.min((P_med, (P_high ** (-2) + P_low ** (-2)) ** (-1/2)), axis = 0)
    
    def computeMdotFortier(self, Rp, Mp):
        """
        Compute the planetesimal accretion rate from Fortier et al (2013) in the absence of migration.

        Rp: Protoplanet location (in AU)
        Mp: Protoplanet mass (in Earth masses)

        return: Planetesimal accretion rate (Earth masses/code time unit)
        """

        disc = self._disc

        # Reduce scope of calculation to planets below the planetesimal isolation mass
        Miso = self.M_iso_pltsml(Rp)
        filter = Mp < Miso

        Rp_grow = Rp[filter]
        Mp_grow = Mp[filter]

        e2 = disc.interp(Rp_grow, disc._planetesimal.e ** 2)
        i2 = disc.interp(Rp_grow, disc._planetesimal.i ** 2)

        Sigma_D = disc.interp(Rp_grow, disc.Sigma_D[2])
        Omega_k = disc.star.Omega_k(Rp_grow) * Omega0
        rH = disc.star.r_Hill(Rp_grow, Mp_grow * Mearth / Msun) * AU

        Mdot = np.zeros_like(Rp)
        Mdot[filter] = Sigma_D * Omega_k * rH ** 2 * self.P_coll(e2, i2, Rp_grow, Mp_grow) / Mearth / Omega0

        return Mdot
    
    # Migration core accretion

    def _R_phys(self, Mp):
        """
        Calculate the embryo radius.

        Mp: Planet mass (Earth masses)
        
        return: R_core in AU
        """

        rho_core = self.rho_core

        return (3/(4*np.pi*rho_core/Msun*AU**3)*Mp*Mearth/Msun)**(1/3)

    def relative_velocity(self, Rp = None):
        """
        Calculate planetesimal velocity relative to the gas.
        
        Rp: Orbital radius at which to evaluate vrel
        
        return: relative velocity
        """
        
        disc = self._disc
        eta = - np.interp(Rp, reduce(disc.R), np.diff(disc.P) / disc.grid.dRc / reduce(disc.midplane_gas_density)) / disc.star.Omega_k(Rp)
        return np.sqrt((disc.star.v_k(Rp) * eta)**2 + np.interp(Rp,reduce(disc.R),disc.gas.viscous_velocity(disc))**2)
 
    def Reynolds(self, Rp, v = None):
        """
        Calculate the Reynolds number.
        
        Rp: Protoplanet radius (in AU)
        v: Relative velocity

        return: Reynolds number
        """

        disc = self._disc

        if v is None:
            v = self.relative_velocity(Rp) 
        
        nu = (disc.visc_mol*Omega0*AU) / (disc.midplane_gas_density*AU**3)
        Re = v * disc.interp(Rp, disc.R_planetesimal / nu)
        return Re
    
    def Mach(self,Rp,v = None):
        """
        Calculate the Mach number.
        
        Rp: Protoplanet radius (in AU)
        Mp: Protoplanet mass (in solar masses)

        return: Mach number
        """

        if v is None:
            v = self.relative_velocity(Rp) 
    
        c_s = self._disc.cs

        Ma = v / self._disc.interp(Rp,c_s)

        return Ma

    def drag_coeff(self, Rp = None):
        """
        Calculate the drag coefficient given by Podolak et al. (1988).
        """

        vrel = self.relative_velocity(Rp)
        Ma = self.Mach(Rp,vrel)
        Re = self.Reynolds(Rp,vrel)
        
        Re = np.where(Re < 1, 1, Re)

        drag_coeff = np.zeros_like(Ma)

        # Calculate the drag coefficient for the different regimes
        # Apply conditions: Ma < 1 and Re < 10^3
        condition = (Ma < 1) & (Re < 1e3) & (Re >= 1)
        drag_coeff[condition] = 6 / np.sqrt(Re[condition])

        # Apply conditions: Ma < 1 and 10^3 < Re < 10^5
        condition = (Ma < 1) & (Re >= 1e3) & (Re < 1e5)
        drag_coeff[condition] = 0.2

        # Apply conditions: Ma < 1 and Re > 10^5
        condition = (Ma < 1) & (Re >= 1e5)
        drag_coeff[condition] = 0.15

        # Apply conditions: Ma > 1 and Re < 1e3
        condition = (Ma >= 1) & (Re < 1e3)
        drag_coeff[condition] = 1.1 - np.log10(Re[condition])/6

        # Apply conditions: Ma > 1 and Re > 10^3
        condition = (Ma >= 1) & (Re >= 1e3)
        drag_coeff[condition] = 0.5

        return drag_coeff

    def R_p_out(self, Rp, Mp):
        """
        Calculate the protoplanet's outer radius.
        
        args:
            Rp: Protoplanet heliocentric radius (in AU)
            Mp: Protoplanet mass (in Earth masses)
        """

        disc = self._disc
        star = disc.star

        rH   = star.r_Hill(Rp, Mp*Mearth/Msun)
        c_s  = disc.interp(Rp, disc.cs)
        M_p  = Mp * Mearth / Msun

        return M_p / (c_s*c_s + (M_p / (0.25 * rH)))
    
    def R_captr_attached(self, Rp, Mp):
        """
        Calculate the protoplanet capture radius according to Valletta & Helled (2021).
        
        args:
            Rp: Protoplanet heliocentric radius (in AU)
            Mp: Protoplanet mass (in Earth masses)
        """

        disc    = self._disc
        star    = disc.star

        rH      = star.r_Hill(Rp, Mp*Mearth/Msun)
        D       = self.drag_coeff(Rp)
        R_pla   = disc.interp(Rp, disc.R_planetesimal)
        rho_pltsml   = disc.rho_pltsml

        # Convert Mp to solar masses for calculations
        Mp_solar_masses     = Mp * Mearth / Msun

        # Planet outer radius
        R0      = self.R_p_out(Rp, Mp)

        # Outer density and pressure of planet envelope (equation 4)
        # Interpolate the disc properties and assume value to be outermomst envelope value
        P0      = disc.interp(Rp, disc.P)
        rho0    = disc.interp(Rp, disc.midplane_density)

        # Calculate alpha parameter (equation 5)
        # Different alpha than viscous
        alpha   = Mp_solar_masses * rho0 / (P0 * R0)
        
        # Calculate rho_star (equation 8) 
        # NOT rho of central star
        rho_star = 2 * R_pla * rho_pltsml / (3 * D * rH)

        # Calculate capture radius (equation 7)
        R_capt  = R0 / (1 + (1/alpha) * np.log10(rho_star/rho0))
    
        return R_capt
    
    def R_captr_detached(self, M_Z, M_HHe, time=1e7):
        """
        Calculate the protoplanet capture radius in the detached phase.
        Applies when M_Z < M_HHe (H+He-dominated envelope).

        args:
            M_Z: Total heavy-element mass (in Earth masses)
            M_HHe: Total H/He mass (in Earth masses)
            time: Time in years (default 1e7). Used to interpolate coefficients
                  between 1e7 and 1e8 years. Values outside this range use nearest set.

        return: Protoplanet capture radius in AU (depends only on M_Z/M_HHe ratio)
        """

        ratio = M_Z / np.maximum(M_HHe, 1e-300)
        ratio = np.clip(ratio, 0, 1.0)

        coeffs_1e7 = np.array([12.80662188, -50.86303789, 382.66267044, -1388.57741163, 1902.60362959])
        coeffs_1e8 = np.array([9.15426162, -6.74548399, 9.40271959, 0, 0])

        if time <= 1e7:
            coeffs = coeffs_1e7
        elif time >= 1e8:
            coeffs = coeffs_1e8
        else:
            log_time = np.log10(time)
            alpha = (log_time - 7.0) / (8.0 - 7.0)
            coeffs = (1 - alpha) * coeffs_1e7 + alpha * coeffs_1e8

        R_capt = np.zeros_like(ratio, dtype=float)
        for i in np.arange(5):
            R_capt += coeffs[i] * ratio**i

        return R_capt * 1.0e9 / AU
    
    def R_capt(self, Rp, Mp, M_Z = None, M_HHe = None, time = 1e7):
        """
        Calculate the protoplanet capture radius.
        
        Rp: Protoplanet radius (in AU)
        Mp: Protoplanet mass (in Earth masses)
        M_Z: Total heavy-element mass (optional, for phase switch)
        M_HHe: Total H/He mass (optional, for phase switch)
        time: Time in years (default 1e7, passed to detached phase calculation)

        return: Protoplanet capture radius
        """

        R_attached = self.R_captr_attached(Rp, Mp)

        if M_Z is None or M_HHe is None:
            self._R_captr = R_attached
            return R_attached

        R_detached = self.R_captr_detached(M_Z, M_HHe, time=time)
        R_captr = np.where(M_Z >= M_HHe, R_attached, R_detached)
        self._R_captr = R_captr

        return R_captr

    def f_g(self, Rp):
        """
        Calculate the surface density scaling factor between our disc and the MMSN.
        
        Rp: Protoplanet location (in AU)
        
        Return: Ratio of disc sigma to MMSN sigma
        """

        mmn_ref = 2400 * (Rp)**(-1.5)
        
        # Calculate normalized gas surface density
        Sigma_G = self._disc.interp(Rp, self._disc.Sigma_G)
        fg = Sigma_G / mmn_ref
        return fg

    def inclination(self, Rp):
        """
        Calculate the planetesimal population inclination.
        
        Parameters:
        Rp: Orbital radius (in AU)
        
        Returns:
        Planetesimal inclination (in radians)
        """

        disc = self._disc

        gamma = disc.interp(Rp,self._stirring) 
        
        # Convert minimum mass solar nebula reference density to g/AU^2
        # Original: 2.4e4 kg/m^2 * (r/AU)^-1
        fg = self.f_g(Rp)
        
        R_pla = disc.interp(Rp, disc.R_planetesimal)  # in AU
        rho_pltsml = self._disc.rho_pltsml

        # Calculate edrag using equation 10
        i0 = 0.23 * ((fg) * (gamma**2) * (R_pla*AU/1e5/1.0) * (rho_pltsml/(3.0)))**(1/3) * (Rp/1.0)**(11/12)
    
        return i0

    def computeAccEff(self, Rp, Mp, dRdt, M_Z = None, M_HHe = None, time = 1e7):
        """
        Calculate the planetesimal accretion efficiency.

        Rp: Protoplanet orbital radius (in AU)
        Mp: Protoplanet mass (in Earth masses)
        dRdt: Protoplanet migration rate
        M_Z: Total heavy-element mass (optional, for phase switch)
        M_HHe: Total H/He mass (optional, for phase switch)
        time: Time in years (default 1e7, for detached phase)

        return: Planetesimal accretion efficiency
        """

        disc = self._disc
        star = disc.star
        
        rH   = star.r_Hill(Rp, Mp*Mearth/Msun)
        h_p = rH/Rp
        R_captr = self.R_capt(Rp, Mp, M_Z=M_Z, M_HHe=M_HHe, time=time)
        R_captr /= rH # capture used instead of physical
        
        i0 = self.inclination(Rp) / h_p

        T_k = (2*np.pi) / star.Omega_k(Rp) # orbital period in 2pi*years

        alpha_pla = 2.5 * np.sqrt(R_captr / (1 + 0.37 * i0*i0 / R_captr))
        beta_pla = 0.79 * (1 + 10 * i0*i0)**(-0.17)

        tau_mig = Rp/np.abs(dRdt) * (h_p**2/T_k)

        b_p = 1 / tau_mig # migration speed

        # Calculate the accretion efficiency
        # Do not allow accretion efficiency to exceed 1.
        acc_eff = np.minimum(1., alpha_pla * b_p ** (beta_pla - 1))
        
        return acc_eff, R_captr

    def computeMdotMigration(self, Rp, Mp, dRdt, M_Z = None, M_HHe = None, time = 1e7):
        """
        Compute the planetesimal accretion rate in the case of migration.
        
        Rp: Protoplanet radius (in AU)
        Mp: Protoplanet mass (in Earth masses)
        M_Z: Total heavy-element mass (optional, for phase switch)
        M_HHe: Total H/He mass (optional, for phase switch)
        time: Time in years (default 1e7, for detached phase)

        return: Planetesimal accretion rate
        """

        disc = self._disc
        Sigma_pla = disc.interp(Rp, disc.Sigma_D[2])
        
        acc_eff = self.computeAccEff(Rp, Mp, dRdt, M_Z=M_Z, M_HHe=M_HHe, time=time)
        R_captr = acc_eff[1]
        acc_eff_Rp = acc_eff[0]

        # Calculate the planetesimal accretion rate
        Mdot = 2 * np.pi * Rp * np.abs(dRdt) * Sigma_pla * acc_eff_Rp / Mearth * AU**2
        self.dRdt = dRdt
        return Mdot
    
    # Old core accretion model
    # Untested and has been superseded by the Fortier et al (2013) model

    def eq_eccentricity_kokubo(self, Rp, Mp, b_tilde = 10):
        """
        Calculate the equilibrium eccentricity of planetesimals based on kokubo et al (2002).
        
        args:
            Rp: Protoplanet location (in AU)
            Mp: Protoplanet mass (in Earth masses)
            b: planetary separation, scaled by hill radius. default 10 from kokubo, could be set based on actual locations

        returns: 
            ndarray: Equilibrium eccentricity of planetesimals
        """

        disc = self._disc
        D = self.drag_coeff(Rp)
        rho_g = disc.interp(Rp,disc.midplane_density)
        rho_pltsml = disc.rho_pltsml
        m_planetesimal = 4/3*np.pi*(disc.interp(Rp, disc.R_planetesimal)*AU)**3*rho_pltsml

        # Calculate equilibirum eccentricity
        e_eq_tilde = 5.6*(m_planetesimal/10**23*(rho_pltsml/2)**2)**(1/15) * (b_tilde/10*D*rho_g/(2*10**-9)*Rp)**(-1/5)
        return e_eq_tilde*(disc.star.r_Hill(Rp,Mp*Mearth/Msun)/Rp)

    def eq_eccentricity_ida2008(self, Rp, r_pltsml = None, eta_ice = 1, iceline = 4):
        """
        Calculate the equilibrium eccentricity of planetesimals based on ida et al (2008).
        This model only uses turbulent stirring
        
        Rp: Protoplanet location (in AU)
        r_pltsml: Planetesimal radius (AU)
        eta_ice: factor for enhancement of solids past iceline in MMSN
        iceline: Ice line location (AU)

        return: equilibrium eccentricity from turbulent excitation
        """

        disc = self._disc
        if r_pltsml is None:
            r_pltsml = disc.interp(Rp, disc.R_planetesimal)
    
        eta_ice_arr = np.ones_like(Rp)
        eta_ice_arr[Rp < iceline] *= eta_ice
        Sigma_D_MMSN = 10*eta_ice_arr*Rp**(-3/2)
        f_d = disc.interp(Rp,disc.Sigma_D.sum(0))/Sigma_D_MMSN # planetesimals included?
        f_g = self.f_g(Rp)
        gamma = disc.interp(Rp,self._stirring)

        rho_pltsml = disc.rho_pltsml
    
        # Calculate equilibirum eccentricities of turbulent stirring vs tidal damping, drag, and collisional damping
        e_tidal = 24 * f_g**0.5 * gamma * ((r_pltsml*AU/1e5/10**3)**3*rho_pltsml/3)**-0.5 * (Rp)**(3/4)
        e_drag = 0.23 * f_g**(1/3) * gamma**(2/3) * (r_pltsml/(10**5/AU)*rho_pltsml/3)**(1/3) * Rp**(11/12)
        e_coll = 3.6 * f_g * (f_d * eta_ice_arr)**-0.5 * gamma * (r_pltsml/(10**5/AU))**0.5 * (rho_pltsml/3)**(5/6) * Rp**(5/4)
       
        min = np.min((e_tidal,e_drag,e_coll),axis=0)
        return min
    
    def eq_eccentricity_makino1993(self, Rp, Mp):
        """
        Compute the equilibrium eccentricity of planetesimals according to Ida and Makino (1993).
        In this model, turbulent stirring is neglected.
        
        args:
            Rp: Protoplanet location (in AU)
            Mp: Protoplanet mass (in Earth masses)
         
        return: 
            array: equilibrium eccentricity from planetesimal-planetesimal or protoplanet-planetesimal interactions
        """

        disc = self._disc
        rho_pltsml = disc.rho_pltsml
        m_planetesimal = 4/3*np.pi*(disc.interp(Rp, disc.R_planetesimal)*AU)**3*rho_pltsml
        
        # Eccentricity excited by planetesimal-planetesimal interaction
        em_mm = 20*(m_planetesimal/1e23)**(-1/15)*(Rp)**(9/20)*(2*m_planetesimal/Msun/(3*disc.star.M))**(1/3)
        
        # Eccentricity excited by protoplanet-planetesimal interaction
        em_Mm = 6*(m_planetesimal/1e23)**(1/18)*(Rp)**(7/24)*((Mp*Mearth/Msun+m_planetesimal/Msun)/(3*disc.star.M))**(1/3)
        return np.max((em_Mm,em_mm),axis=0)
    
    def compute_v_ran(self, Rp, Mp):
        """
        Calculate the relative velocity between the protoplanet and the planetesimals.
        
        Rp: Protoplanet location (in AU)
        Mp: Protoplanet mass (in Earth masses)

        return: Relative velocity
        """

        disc = self._disc
        r_H = disc.star.r_Hill(Rp,Mp*Mearth/Msun)
        eq_run = eq_oli = np.zeros_like(Rp)

        # Find equilibrium eccentrities according to runaway growth model
        if self._run_model == 'ida2008':
            eq_run = self.eq_eccentricity_ida2008(Rp)
        elif self._run_model == 'makino1993':
            eq_run = self.eq_eccentricity_makino1993(Rp, Mp)
        
        # Find equilibrium eccentricities according to oligarchic growth model
        if self._olig_model == 'kokubo2002':
            eq_oli = self.eq_eccentricity_kokubo(Rp, Mp)
        elif self._olig_model == 'makino1993':
            if self._run_model == 'makino1993':
                pass #save time if Makino is used for both
            else:
                eq_oli = self.eq_eccentricity_makino1993(Rp, Mp)
            
        # Combine oligarchic and runaway growth into one array and calculate dispersion velocity
        v_disp = np.max((eq_oli,eq_run),axis=0) * disc.star.v_k(Rp)

        return v_disp

    def planetesimal_iso_mass(self, Rp):
        """
        Planetesimal isolation mass for model in which neither planetesimals nor protoplanets are migrating.
        
        Rp: Protoplanet location (AU)

        return: Planetesimal isolation mass (Earth masses)
        """

        return 0.1*(self._disc.interp(Rp,self._disc.Sigma_D[2])/5)**1.5 * (Rp)**3 * (self._disc.star.M)**-0.5

    def computeMdotTwoPhase(self, Rp, Mp, dRdt=None):
        """
        Compute the planetesimal accretion rate in the absence of migration.
        
        Rp: Protoplanet radius (in AU)
        Mp: Protoplanet mass (in Earth masses)

        return: Planetesimal accretion rate (Earth masses/code time unit)
        """

        disc = self._disc

        disc._v_drift = np.concatenate((disc.v_drift,[np.zeros_like(disc.v_drift[1])]))

        # Reduce scope of calculation to planets below the isolation mass
        m_iso = self.planetesimal_iso_mass(Rp)
        filter = Mp < m_iso
        Rp_grow = Rp[filter]
        Mp_grow = Mp[filter]
        Sigma_pla = disc.interp(Rp_grow,disc.Sigma_D[2])

        r_physical = self._R_phys(Mp_grow)

        # Obtain random velocity between protoplanet and planetesimals
        v_rel = self.compute_v_ran(Rp_grow,Mp_grow)

        v_esc_sqrd = 2*Mp_grow*Mearth/Msun/r_physical
        Mdot = np.zeros_like(Rp,dtype=np.float64)
        
        # Compute Mdot from random velocity
        Mdot[filter] = 2*(np.pi*disc.star.Omega_k(Rp_grow)*Sigma_pla/Msun*AU**2*r_physical**2*(v_esc_sqrd/v_rel**2))*Msun/Mearth
        
        return Mdot

    def update(self):
        """Update internal quantities after the disc has evolved."""
        pass

################################################################################
# Migration
################################################################################

def _GK(p):
    gk0 = 16/25.

    f1 = gk0*p**1.5
    f2 = 1 - (1-gk0)*p**-(8/3.)

    return np.where(p < 1, f1, f2)

def _F(p):
    return 1 / (1 + (p/1.3)**2)

# Linblad torque
def _linblad(alpha, beta):
    return -2.5 - 1.7*beta + 0.1*alpha

# Linear co-rotation torques
def _cr_baro(alpha):
    return 0.7 * (1.5 - alpha)

def _cr_entr(alpha, beta, gamma):
    return (2.2 - 1.4/gamma) * (beta - (gamma-1)*alpha)

# Non-linear horse-shoe drag torques
def _hs_baro(alpha):
    return 1.1 * (1.5 - alpha)

def _hs_entr(alpha, beta, gamma):
    return 7.9 *(beta - (gamma-1)*alpha) / gamma

_k0 = np.sqrt(28 / (45 * np.pi))
def _K(p):
    return _GK(p/_k0)

_g0 = np.sqrt(8 / (45 * np.pi))
def _G(p):
    return _GK(p/_g0)



class TypeIMigration(object):
    """
    Type 1 Migration model of planets by Paardekooper et al (2011).

    Only implemented for sofenting the default softening parameter b/h=0.4

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
        M     : central mass, default = 1
    
    Note: 
        This modified version of Paardekooper's model assumes that 
        disk wind alpha parameter has a similar affect on type 1 migration 
        as viscous alpha.
    """

    def __init__(self, disc, gamma=1.4):
        self._gamma = gamma

        #Tabulate gamma_eff to avoid underflow/overflow
        self._Q_tab = np.logspace(-2, 2, 100)
        self._gamma_eff_tab = self._gamma_eff(self._Q_tab)

        self.set_disc(disc)

    def ASCII_header(self):
        return '# {} gamma: {}'.format(self.__class__.__name__,
                                       self._gamma)
    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "gamma" : "{}".format(self._gamma) }

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def update(self):
        """Update internal quantities after the disc has evolved"""
        disc = self._disc
        
        lgR = np.log(disc.R)
        # Horibble hack to smooth out artifacts?
        _lgSig = ispline(lgR, np.log(disc.Sigma))
        _lgT   = ispline(lgR, np.log(disc.T))

        self._dlgSig = _lgSig.derivative(1)
        self._dlgT   = _lgT.derivative(1)

    # Fitting functions

    def _gamma_eff(self, Q):
        """Effective adiabatic index"""
        Qg = Q*self._gamma
        Qg2 = Qg*Qg
        gm1 = self._gamma-1
        
        f1 = 2*np.sqrt((Qg2 + 1)**2 - 16*Q*Q*gm1)
        f2 = 2*Qg2-2

        return 2*Qg / (Qg + 0.5*np.sqrt(f1 + f2))

    def gamma_eff_tab(self, Q):
        """Effective adiabatic index, tabulated"""
        return np.interp(Q, self._Q_tab, self._gamma_eff_tab)
        
    def compute_torque(self, Rp, Mp):
        """Compute the torques acting on a planet driving Type I migration"""
        disc = self._disc
        star = disc.star
        
        # Interpolate the disc properties
        lgR = np.log(Rp)
        alpha = -self._dlgSig(lgR)
        beta  = -self._dlgT(lgR)

        h     = disc.interp(Rp, disc.H) / Rp
        Sigma = disc.interp(Rp, disc.Sigma)
        nu_SS = disc.interp(Rp, disc.nu)
        nu    = disc.interp(Rp, disc.nu)
        Pr    = disc.interp(Rp, disc.Pr)

        Om_k = star.Omega_k(Rp)
        # Include disk wind contribution to temperature
        Xi = nu_SS/Pr* (1 + disc._gas._psi/3.)
        Q = 2*Xi/(3*h*h*h*Rp*Rp*Om_k)
        g_eff = self.gamma_eff_tab(Q)
        
        q_h = (Mp*Mearth/(star.M*Msun)) / h

        jp = Om_k*Rp*Rp
        Om_kr_2 = jp*jp

        # Convert from g cm^-2 AU**4 Omega0**2 to Mearth AU**2 Omega0**2
        norm  = q_h*q_h*Sigma*Om_kr_2 / g_eff
        norm *= AU**2/Mearth
        
        # Compute the scaling factors
        k = jp / (2*np.pi * nu_SS)
        kXi  = jp / (2*np.pi * Xi)
        x = (1.1 / g_eff**0.25) * np.sqrt(q_h)

        pnu = 2*np.sqrt(k*x*x*x)/3
        pXi  = 2*np.sqrt(kXi * x*x*x) / 3

        Fnu, Gnu, Knu = _F(pnu), _G(pnu), _K(pnu)
        FXi, GXi, KXi = _F(pXi), _G(pXi), _K(pXi)
        
        torque = (_linblad(alpha, beta) +
                  _hs_baro(alpha) * Fnu * Gnu +
                  _cr_baro(alpha) * (1 - Knu) +
                  _hs_entr(alpha, beta, g_eff) * Fnu * FXi * np.sqrt(Gnu*GXi) +
                  _cr_entr(alpha, beta, g_eff) * np.sqrt((1-Knu)*(1-KXi)))

        return norm*torque

    def migration_rate(self, Rp, Mp):
        """Migration rate, dRdt, of the planet according to Paardekooper et al (2011)"""
        J = Mp*Rp*self._disc.star.v_k(Rp)
        return 2 * (Rp/J) * self.compute_torque(Rp, Mp)
    
    def __call__(self, planets):
        """Migration rate, dRdt, of the planet"""
        return self.migration_rate(planets.R, planets.M)
    

    
class TypeIIMigration(object):
    """
    Giant planet migration. Uses relation of Baruteau et al (2014). 
    Note, for disk winds, assumes disk wind alpha parameter has a 
    similar affect on type 1 migration as viscous alpha.
    """

    def __init__(self, disc):
        self._disc = disc

    def ASCII_header(self):
        """Generate ASCII header string"""
        return '# {}'.format(self.__class__.__name__)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, {}

    def set_disc(self, disc):
        self._disc = disc
        self.update()

    def migration_rate(self, Rp, Mp):
        """Migration rate, dR/dt, of the planet according to Baruteau et. al 2014"""
        disc = self._disc
        
        Sigma = disc.interp(Rp, disc.Sigma)
        nu    = disc.interp(Rp, disc.nu) * (1 + disc._gas._psi)

        Sigma *= AU**2/Mearth

        t_mig = Rp*Rp/nu * np.maximum(Mp /(4*np.pi*Sigma*Rp*Rp), 1)

        return - Rp / t_mig

    def __call__(self, planets):
        """Migration rate, dRdt, of the planet"""
        return self.migration_rate(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        pass

################################################################################
# Combined models
################################################################################
    
class PlanetMigration(object):
    """
    Migration by Type I and Type II with a switch based on the Crida &
    Morbidelli (2007) gap depth criterion.

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
        winds : Whether the disk includes disk winds, default=False
    
    Note:
        Originally, this migration model was based of Bitsch et. al (2015).
        Due to this model being incorrect for, and a lack of migration research 
        in, low viscosity disks (as seen with disk winds), this version assumes 
        that the disk wind alpha affects planet migration similarly to 
        viscous alpha.
    """
    
    def __init__(self, disc, gamma=1.4, winds=False):
        if not winds:
            # ViscousEvolution classes do not assign a psi value,
            # so assign one here to not error in migration calculations.
            disc._gas._psi = 0

        self._typeI  = TypeIMigration(disc, gamma=gamma)
        self._typeII = TypeIIMigration(disc)
        self._disc = disc

    def ASCII_header(self):
        head = '# {} \n#\t{}\n#\t{}'.format(self.__class__.__name__,
                                            self._typeI.ASCII_header()[1:],
                                            self._typeII.ASCII_header()[1:])
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, dict([self._typeI.HDF5_attributes(),
                                              self._typeII.HDF5_attributes()])

    def set_disc(self, disc):
        self._typeI.set_disc(disc)
        self._typeII.set_disc(disc)

        self._disc = disc

    def migration_rate(self, Rp, Mp):
        """Compute migration rate according to Bitsch et. al (2015)"""
        disc = self._disc
        star = disc.star
        
        vr_I  = self._typeI.migration_rate(Rp, Mp)
        vr_II = self._typeII.migration_rate(Rp, Mp)

        Me = Mp*Mearth/Msun
        q = Me / star.M
        rH = star.r_Hill(Rp, Mp)
        #nu = disc.interp(Rp, disc.nu) * (1 + disc._gas._psi)
        nu = disc.interp(Rp, disc.nu)

        H  = disc.interp(Rp, disc.H)

        Re = Rp * star.v_k(Rp) / nu

        P = np.maximum(0.75*H/rH + 50/(q*Re), 0.541)

        fP = np.where(P < 2.4646, 0.25*(P-0.541), 1 - np.exp(-P**0.75/3))

        return fP*vr_I + (1-fP)*vr_II
        # For testing:
        #return fP*vr_I*0 + (1-fP)*vr_II

    def __call__(self, planets):
        """Compute migration rate"""
        return self.migration_rate(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        self._typeI.update()
        self._typeII.update()
        
    
        
class Bitsch2015Model(object):
    """
    Pebble accretion + Gas accretion planet formation model based on Bisch et al (2015).

    The model is composed of the Hill branch pebble accretion along with gas envelope accretion.

    args:
        disc     : accretion disc model
        pb_gas_f : fraction of pebble accretion rate that arrives as gas, default = 0.1
        f_plt    : ratio between the embryo birth mass and the planetesimal birth mass, default = 400
        migrate  : Whether to include migration, default = True
        pebble_acc : Whether to include pebble accretion, default = True
        planetesimal_acc_migrate: function to compute accretion rate during migration, default = True
        planetesimal_acc_insitu: function to compute accretion rate in-situ, default = True
        gas_acc: model for gas accretion
        winds    : Whether the disk includes disk winds, default = False
        rho_core : core density [g cm^-3], default = 5.5
        **kwargs : additional arguments passed to GasAccretion object
    """

    def __init__(self, disc, pb_gas_f = 0.1, f_plt = 400, migrate = True, pebble_acc = True, planetesimal_acc_migrate = True, planetesimal_acc_insitu = True, gas_acc = True, winds = False, rho_core = 5.5, **kwargs):

        self._f_gas = pb_gas_f
        self._f_plt = f_plt
        self._disc = disc

        self._gas_acc = None
        if gas_acc:
            self._gas_acc = GasAccretion(disc, rho_core = rho_core, **kwargs)

        self._peb_acc = None
        if pebble_acc:
            self._peb_acc = PebbleAccretion(disc)

        self._pla_acc = None
        if planetesimal_acc_migrate or planetesimal_acc_insitu:
            self._pla_acc = PlanetesimalAccretion(disc, Mdot_migrate = planetesimal_acc_migrate, Mdot_insitu = planetesimal_acc_insitu, rho_core = rho_core)

        self._migrate = None
        if migrate:
            self._migrate = PlanetMigration(disc, winds = winds)

    def ASCII_header(self):
        """header"""
        head ='# {} pb_gas_f: {}, migrate: {}\n'.format(self.__class__.__name__,
                                                        self._f_gas,
                                                        bool(self._migrate))
        head += self._gas_acc.ASCII_header()
        if self._peb_acc:
            head += '\n' + self._peb_acc.ASCII_header()
        if self._migrate:
            head += '\n' + self._migrate.ASCII_header()
        return head

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        head = {
            "pb_gas_f": "{}".format(self._f_gas),
            "migrate": "{}".format(self._migrate)
        }
        head.update(self._gas_acc.HDF5_attributes()[1])
        if self._peb_acc:
            head.update(self._peb_acc.HDF5_attributes()[1])
        if self._migrate:
            head.update(self._migrate.HDF5_attributes()[1])
        return self.__class__.__name__, head

    def set_disc(self, disc):
        """Set up the current disc model"""
        if self._gas_acc:
            self._gas_acc.set_disc(disc)

        if self._peb_acc:
            self._peb_acc.set_disc(disc)

        if self._pla_acc:
            self._pla_acc.set_disc(disc)

        if self._migrate:
            self._migrate.set_disc(disc)

        self._disc = disc
            
    def update(self):
        """Update internal quantities after the disc has evolved"""
        if self._gas_acc:
            self._gas_acc.update()
        if self._peb_acc:
            self._peb_acc.update()
        if self._pla_acc:
            self._pla_acc.update()
        if self._migrate:
            self._migrate.update()

    def M_on_Lorek(self, Rp):
        """
        Compute the pebble accretion onset mass (Lorek et al 2022).

        Rp: Protoplanet location (in AU)

        return: Pebble accretion onset mass (in Earth masses)
        """

        disc = self._disc

        Mstar = disc.star.M
        Lstar = disc.star.L
        v_frag = disc._uf_0 * (AU * Omega0)
        alpha = disc.alpha
        cs = disc.interp(Rp, disc.cs) * (AU * Omega0)
        epsilon_g = 0.5
        d2g = disc.interp(Rp, disc.integ_dust_frac)
        Omega_k = disc.star.Omega_k(Rp) * Omega0
        rho_g = disc.interp(Rp, disc.midplane_gas_density)
        dP = disc.interp(Rp, disc.dP_dR)

        eta = -1 * dP / (2 * Omega_k ** 2 * Rp * AU * rho_g)

        # Fragmentation limited Stokes number
        tau_s_frag = v_frag ** 2 / (2 * alpha * cs ** 2)

        # Drift limited Stokes number
        tau_s_drift = 3 * np.sqrt(np.pi) / 4 * epsilon_g * d2g / eta

        tau_s = np.minimum(tau_s_frag, tau_s_drift)

        return 4.871e-7 * (tau_s / 0.01) * Mstar ** (-17/7) * Lstar ** (6/7) * Rp ** (12/7)
    
    def M_tr_Lorek(self, Rp):
        """
        Compute the pebble accretion transition mass (Lorek et al 2022).

        Rp: Protoplanet location (in AU)

        return: Pebble accretion transition mass (in Earth masses)
        """

        disc = self._disc

        Mstar = disc.star.M
        Lstar = disc.star.L

        return 1.125e-4 * Mstar ** (-17/7) * Lstar ** (6/7) * Rp ** (12/7)
    
    def M_iso_Lorek(self, Rp):
        """
        Compute the pebble isolation mass (Lorek et al 2022).

        Rp: Protoplanet location (in AU)

        return: Pebble isolation mass (in Earth masses)
        """

        disc = self._disc

        h = disc.interp(Rp, disc.h)
        alpha = disc.alpha
        dlnP = Rp * AU / disc.interp(Rp, disc.P) * disc.interp(Rp, disc.dP_dR)

        return 25 * (h / 0.05) ** 3 * (0.34 * (-3 / np.log10(alpha)) ** 4 + 0.66 ) * (1 - (dlnP + 2.5) / 6)
    
    def M_birth_embryo(self, Rp):
        """
        Computes the streaming instability birth mass of embryos in Earth masses (equation 14 from Liu et al 2020).

        Rp: Protoplanet location (in AU)

        return: Streaming instability birth mass (in Earth masses)
        """

        disc = self._disc
        
        rho_g = disc.interp(Rp, disc.midplane_gas_density)
        Omega_k = disc.star.Omega_k(Rp)
        Z = disc.dust_frac_SI
        Mstar = disc.star.M
        h = disc.interp(Rp, disc.h)

        gamma = 4 * np.pi * G * rho_g / (Omega_k ** 2) * (AU ** 3 / Msun) # self gravity term

        M_birth_pltsml = 5e-6 * (Z / 0.02) ** 0.5 * (gamma * np.pi) ** 1.5 * (h / 0.05) ** 3 * (Mstar / 0.1)

        return self._f_plt * M_birth_pltsml
    
    def insert_new_planet(self, t, R, M, planets):
        """
        Set the initial mass of the planets.

        args:
            t : current time
            R : AU, formation locations
            M : initial mass of the planet
            planets : planets object to add planets to
        """

        self._use_SI = str(M).upper() == 'SI'
        self._use_PA = str(M).upper() == 'PA'
        self._use_TR = str(M).upper() == 'TR'

        # Set initial core mass

        if self._use_SI:
            Mc = self.M_birth_embryo(R)

        elif self._use_PA:
            Mc = self.M_on_Lorek(R)

        elif self._use_TR:
            Mc = self.M_tr_Lorek(R)

        else:
            Mc = M

        Me = 0.0

        # Set initial chemistry

        if planets.chem:
            Xs, Xg = self._compute_chem(R)

        else:
            Xs, Xg = None, None
            
        planets.add_planet(t, R, Mc, Me, Xs, Xg)

    def _compute_chem(self, R_p):
        disc = self._disc
        chem = disc.chem
        
        Xs = []
        Xg = []

        eps_dust = np.maximum(disc.interp(R_p, disc.dust_frac[:2].sum(0)), 1e-300)

        for spec in chem:
            Xs_i, Xg_i = chem.ice[spec], chem.gas[spec]
            Xs.append(disc.interp(R_p, Xs_i) / eps_dust)
            Xg.append(disc.interp(R_p, Xg_i))

        return np.array(Xs), np.array(Xg)

    def _compute_chem_planetesimal(self, R_p):
        disc = self._disc
        chem = disc.chem

        Xs_pla = []

        for spec in chem:
            if self._pla_acc:
                Xs_pla.append(disc.interp(R_p, disc._planetesimal.ice_abund[spec]) / np.maximum(disc.interp(R_p, disc.dust_frac[2]), 1e-300))

            else:
                Xs_pla = np.zeros_like(R_p)

        return np.array(Xs_pla)

    def _growth_rates(self, R_p, M_core, M_env, M_Z, M_HHe):
        """
        Compute every term of the planet growth/migration ODE at a given state, and the masks that decide which terms are actually active.

        args:
            R_p, M_core, M_env : planet state (AU, Mearth, Mearth)
            M_Z, M_HHe         : heavy-element / H-He mass, for the planetesimal accretion phase switch

        returns:
            Rdot                        : migration rate [AU / code time]
            Mdot_gas                    : gas envelope accretion rate
            Mdot_pebble_core            : pebble accretion rate landing on the core
            Mdot_pebble_env             : pebble accretion rate landing on the envelope
            Mdot_planetesimal_insitu    : Fortier et al 2013 rate, active where it exceeds the migration rate
            Mdot_planetesimal_migration : rate accreted while migrating, active where it exceeds the in-situ rate
            f_pla                       : fraction of planetesimal accretion assigned to the envelope
            Mcdot, Medot                : totals actually integrated into M_core, M_env
        """

        Rmin = self._disc.R[0]

        # Migration
        Rdot = np.zeros_like(R_p)

        if self._migrate:
            Rdot = self._migrate.migration_rate(R_p, M_core + M_env)

        # Gas accretion
        Mdot_gas = np.zeros_like(R_p)

        if self._gas_acc:
            Mdot_gas = self._gas_acc.computeMdot(R_p, M_core, M_env)

        # Pebble accretion
        f = self._f_gas # fraction of pebble accretion assigned to the envelope
        Mdot_pebble_core = np.zeros_like(R_p)
        Mdot_pebble_env  = np.zeros_like(R_p)

        if self._peb_acc:
            Mdot_pebble = self._peb_acc.computeMdot(R_p, M_core + M_env)
            Mdot_pebble_core = Mdot_pebble * (1 - f)
            Mdot_pebble_env  = Mdot_pebble * f

        # Planetesimal accretion
        Mdot_planetesimal_insitu    = np.zeros_like(R_p)
        Mdot_planetesimal_migration = np.zeros_like(R_p)
        f_pla = np.where(M_env >= 1.0, 1.0, 0.0) # fraction of planetesimal accretion assigned to the envelope
        use_migration = np.zeros_like(R_p, dtype = bool)

        if self._pla_acc:
            Mdot_insitu = np.zeros_like(R_p)
            if self._pla_acc._Mdot_insitu:
                Mdot_insitu = self._pla_acc.computeMdotFortier(R_p, M_core + M_env)

            Mdot_migration = np.zeros_like(R_p)
            if self._pla_acc._Mdot_migrate:
                Mdot_migration = self._pla_acc.computeMdotMigration(R_p, M_core + M_env, Rdot, M_Z = M_Z, M_HHe = M_HHe)

            # Only use the planetesimal accretion channel that is larger for each planet
            use_migration = Mdot_migration > Mdot_insitu

            Mdot_planetesimal_insitu    = np.where(~use_migration, Mdot_insitu, 0.0)
            Mdot_planetesimal_migration = np.where(use_migration, Mdot_migration, 0.0)

        # Planets that have migrated to the disc's inner edge stop growing
        accreted = R_p <= Rmin
        Rdot                        = np.where(accreted, 0, Rdot)
        Mdot_gas                    = np.where(accreted, 0, Mdot_gas)
        Mdot_pebble_core            = np.where(accreted, 0, Mdot_pebble_core)
        Mdot_pebble_env             = np.where(accreted, 0, Mdot_pebble_env)
        Mdot_planetesimal_insitu    = np.where(accreted, 0, Mdot_planetesimal_insitu)
        Mdot_planetesimal_migration = np.where(accreted, 0, Mdot_planetesimal_migration)

        Mdot_planetesimal = Mdot_planetesimal_insitu + Mdot_planetesimal_migration

        Mcdot = Mdot_pebble_core + Mdot_planetesimal * (1 - f_pla)
        Medot = Mdot_gas + Mdot_pebble_env + Mdot_planetesimal * f_pla

        return {
            "Rdot": Rdot,
            "Mdot_gas": Mdot_gas,
            "Mdot_pebble_core": Mdot_pebble_core,
            "Mdot_pebble_env": Mdot_pebble_env,
            "Mdot_planetesimal_insitu": Mdot_planetesimal_insitu,
            "Mdot_planetesimal_migration": Mdot_planetesimal_migration,
            "f_pla": f_pla,
            "Mcdot": Mcdot,
            "Medot": Medot}

    def integrate(self, dt, planets):
        """
        Update the planet masses and radii.

        args:
            dt      : Time to integrate for
            planets : Planets container
        """

        if planets.N == 0:
            return

        self.update()

        chem = planets.chem

        N = planets.N

        def f_integ(_, y):
            R_p    = y[   :  N]
            M_core = np.where(y[N  :2*N] < 0, 0, y[N  :2*N])
            M_env  = np.where(y[2*N:3*N] < 0, 0, y[2*N:3*N]) # avoid negative envelope masses

            # Extract M_Z and M_HHe for phase switch
            if chem:
                Nspec = (len(y) - 3*N) // (2*N)
                Chem_core = y[3*N : 3*N + Nspec*N].reshape(Nspec, N)
                Chem_env = y[3*N + Nspec*N : 3*N + 2*Nspec*N].reshape(Nspec, N)
                M_Z = Chem_core.sum(0) + Chem_env.sum(0)
                M_HHe = M_core + M_env - M_Z

            else:
                M_Z = M_core
                M_HHe = M_env

            rates = self._growth_rates(R_p, M_core, M_env, M_Z, M_HHe)

            dydt = np.empty_like(y)
            dydt[:N]      = rates["Rdot"]
            dydt[N:2*N]   = rates["Mcdot"]
            dydt[2*N:3*N] = rates["Medot"]

            if chem:
                Xs, Xg = self._compute_chem(R_p)
                Xs_pla = self._compute_chem_planetesimal(R_p)
                Nspec = Xs.shape[0]

                Mdot_planetesimal = rates["Mdot_planetesimal_insitu"] + rates["Mdot_planetesimal_migration"]
                f_pla = rates["f_pla"]
                Mg = np.maximum(rates["Mdot_gas"], 0)

                dydt[ 3       *N:(3+  Nspec)*N] = (rates["Mdot_pebble_core"] * Xs + Mdot_planetesimal * (1 - f_pla) * Xs_pla).ravel()
                dydt[(3+Nspec)*N:(3+2*Nspec)*N] = (rates["Mdot_pebble_env"] * Xs + Mg * Xg + Mdot_planetesimal * f_pla * Xs_pla).ravel()

            return dydt

        integ = ode(f_integ).set_integrator('dopri5', rtol = 1e-5, atol = 1e-5)

        if chem:
            Chem_core = (planets.M_core * planets.X_core).flat
            Chem_env  = (planets.M_env  * planets.X_env).flat
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env, Chem_core, Chem_env])

        else:
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env])

        integ.set_initial_value(X0, 0)

        #print(f"Before integration: R: {planets.R}, M_core: {planets.M_core}, M_env: {planets.M_env}")  # Debugging print
        integ.integrate(dt)
        #print(f"After integration: R: {integ.y[:N]}, M_core: {integ.y[N:2*N]}, M_env: {integ.y[2*N:3*N]}")  # Debugging print

        # Compute the fraction of the core / envelope that was accreted in solids

        planets.R = integ.y[:N]
        planets.M_core = integ.y[N:2*N]
        planets.M_env  = integ.y[2*N:3*N]

        if chem:
            Ns = np.prod(planets.X_core.shape)
            Xc = integ.y[3*N   :3*N  +Ns].reshape(-1, N)
            Xe = integ.y[3*N+Ns:3*N+2*Ns].reshape(-1, N)
            planets.X_core = Xc / np.maximum(planets.M_core, 1e-300)
            planets.X_env  = Xe / np.maximum(planets.M_env, 1e-300)

            M_Z   = (planets.X_core * planets.M_core).sum(0) + (planets.X_env * planets.M_env).sum(0)
            M_HHe = planets.M_core + planets.M_env - M_Z

        else:
            M_Z   = planets.M_core
            M_HHe = planets.M_env

        # Record the exact rates that produced this step, for diagnostics/output.
        self.rates = self._growth_rates(planets.R, planets.M_core, planets.M_env, M_Z, M_HHe)

    def dump(self, filename, time, planets):
        """Write out the planet info"""

        # First get the header info.
        with open(filename, 'w') as f:
            head = self.ASCII_header()
            f.write(head+'\n')
            print('# time: {}yr\n'.format(time / (2 * np.pi)))

            head = '# R M_core M_env t_form'
            if planets.chem:
                chem = self._disc.chem
                for k in chem.gas:
                    head += ' c{}'.format(k)
                for k in chem.ice:
                    head += ' e{}'.format(k)
            f.write(head+'\n')

            for p in planets:
                f.write('{} {} {} {}'.format(p.R, p.M_core, p.M_env, 
                                             p.t_form / (2 * np.pi)))
                if planets.chem:
                    for Xi in p.X_core:
                        f.write(' {}'.format(Xi))
                    for Xi in p.X_env:
                        f.write(' {}'.format(Xi))
                f.write('\n')
                        

            
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .eos import LocallyIsothermalEOS, IrradiatedEOS
    from .star import SimpleStar
    from .grid import Grid
    from .dust import FixedSizeDust

    GM = 1.
    cs0 = (1/30.) 
    q = -0.25
    Mdot = 1e-9
    alpha = 1e-3

    Mdot *= Msun / (2*np.pi)
    Mdot /= AU**2

    Rin = 0.01
    Rout = 5e2
    Rd = 100.

    t0 = (2*np.pi)

    star = SimpleStar()
    
    grid = Grid(0.01, 1000, 1000, spacing='log')
    eos = LocallyIsothermalEOS(star, cs0, q, alpha)
    eos.set_grid(grid)
    Sigma =  (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)
    if 1:
        eos = IrradiatedEOS(star, alpha, tol=1e-3, accrete=False)     
        eos.set_grid(grid)
        eos.update(0, Sigma)
        
        # Now do a new guess for the surface density and initial eos.
        Sigma = (Mdot / (3 * np.pi * eos.nu))*np.exp(-grid.Rc/Rd)

        eos = IrradiatedEOS(star, alpha, tol=1e-3)
        eos.set_grid(grid)
        eos.update(0, Sigma)
    disc = FixedSizeDust(grid, star, eos, 1e-2, 1, Sigma)
    R = disc.R

    # Test the migration rate calculation

    migI  = TypeIMigration(disc)
    migII = TypeIIMigration(disc)

    migCrida = PlanetMigration(disc)

    Rp = [1,5,25,100]
    M_p = np.logspace(-3, 4.0, 100)
    
    planets = Planets()
    for Mi in M_p:
        planets.add_planet(0, 1, Mi, 0)
    
    plt.subplot(211)
    for Ri in Rp:
        planets.R[:] = Ri
        Ri = Ri * np.ones_like(M_p)
        l, = plt.loglog(M_p, -Ri/migCrida(planets)/t0)
        plt.loglog(M_p, -Ri/migI(planets)/t0,  c=l.get_color(), ls='--')
        plt.loglog(M_p,  Ri/migI(planets)/t0,  c=l.get_color(), ls='-.')
        plt.loglog(M_p, -Ri/migII(planets)/t0, c=l.get_color(), ls=':')

    plt.xlabel('$M\,[M_\oplus]$')
    plt.ylabel('$t_\mathrm{mig}\,[yr]$')

    Rp = np.logspace(-0.5,2,100)
    planets.R[:] = Rp
    plt.subplot(212)
    for Mi in [1, 3, 10, 30]:
        planets.M_core[:] = Mi
        l, =plt.loglog(Rp, -Rp/migCrida(planets)/t0)
        plt.loglog(Rp, -Rp/migI(planets)/t0,  c=l.get_color(), ls='--')
        plt.loglog(Rp,  Rp/migI(planets)/t0,  c=l.get_color(), ls='-.')
        plt.loglog(Rp, -Rp/migII(planets)/t0, c=l.get_color(), ls=':')
    plt.xlabel('$R\,[AU]$')
    plt.ylabel('$t_\mathrm{mig}\,[yr]$')

    # Test the growth models

    # Set up some planet mass / envelope ratios
    M_p = planets.M
    planets.M_core = np.minimum(20, 0.9*M_p)
    planets.M_env  = M_p - planets.M_core

    #Sigma = 1700 * R**-1.5
    Rp = [0.5, 5., 50.]
    
    PebAcc = PebbleAccretion(disc)
    GasAcc = GasAccretion(disc)

    plt.figure()
    for Ri in Rp:
        planets.R[:] = Ri
        l, = plt.loglog(M_p, M_p/PebAcc(planets)/t0)
        plt.loglog(M_p, M_p/GasAcc(planets)/t0,
                   c=l.get_color(), ls='--')

    plt.xlabel('$M\,[M_\oplus]$')
    plt.ylabel('$t_\mathrm{grow}\,[yr]$')

    # Growth tracks

    plt.figure()

    planet_model = Bitsch2015Model(disc, pb_gas_f=0.0)

    times = np.logspace(0, 7, 200)
    Rp  = np.array(Rp)

    planets = Planets()
    for Ri in Rp:
        planet_model.insert_new_planet(0, Ri, M_p, planets)

    print(planets.R)
    print(planets.M_core)
    print(planets.M_env)
        
    Rs, Mcs, Mes, = [], [], []
    t = 0
    for ti in times:
        ti *= t0
        planet_model.integrate(ti-t, planets)
        Rs.append(planets.R.copy())
        Mcs.append(planets.M_core.copy())
        Mes.append(planets.M_env.copy())
        t = ti

    Rs, Mcs, Mes = [ np.array(X) for X in [Rs, Mcs, Mes]]
        
    ax =plt.subplot(311)
    plt.loglog(times, Mcs)
    plt.ylabel('$M_\mathrm{core}\,[M_\oplus]$')
    plt.ylim(ymax=1e3)

    plt.subplot(312, sharex=ax)
    plt.loglog(times, Mes/317.8)
    plt.ylabel('$M_\mathrm{env}\,[M_J]$')

    plt.subplot(313, sharex=ax)
    plt.loglog(times, Rs)
    plt.ylabel('$R\,[\mathrm{au}]$')
    plt.ylim(Rin, Rout)
    
    plt.xlabel('$t\,[yr]$')
    plt.show()
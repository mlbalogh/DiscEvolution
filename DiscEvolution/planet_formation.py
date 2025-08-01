from __future__ import print_function
import numpy as np
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
    """Data for growing planets.

    Holds the location, core & envelope mass, and composition of growing
    planets.

    args:
        Nchem : number of chemical species to track, default = None
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
    """Gas giant accretion model of Bitsch et al (2015).

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
           rho_core  : core density [g cm^-3], default=5.5
    """
    def __init__(self, disc, f_max=0.8,
                 f_py=0.2, kappa_env=0.05, rho_core=5.5):

        # General properties
        self._fmax = f_max
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
        """Compute gas accretion rate.

        args:
            Rp     : radius, AU
            M_core : Core mass, Mearth
            M_env  : Envelope mass, Mearth

        returns:
            Mdot : accretion rate in Mearth per Omega0**-1
        """
        # Cache data:
        Mp = M_core + M_env
        
        # Piso & Youdin (2014) accretion rate:
        T81 = self._disc.interp(Rp, self._disc.T)/81
        Mdot_PY = self._fPiso * T81**-0.5 * M_core**(11/3.) / M_env
        
        # Machida+ (2010) accretion rate
        star = self._disc.star
        rH = star.r_Hill(Rp, Mp*Mearth/Msun)

        Sig = self._disc.interp(Rp, self._disc.Sigma_G)
        H   = self._disc.interp(Rp, self._disc.H)
        nu  = self._disc.interp(Rp, self._disc.nu)

        Om_k = star.Omega_k(Rp)
        
        # Accretion rate is the minimum of two branches, meeting at
        # rH/H ~ 0.3
        f = np.minimum(0.83 * (rH/H)**4.5, 0.14)

        
        # Convert to Mearth / AU**2
        Sig /= Mearth/AU**2

        Mdot_Machida = f * Om_k * Sig * H*H

        Mdot = np.where(M_core > M_env, Mdot_PY, Mdot_Machida)

        return np.minimum(Mdot, self._fmax * 3*np.pi*Sig*nu)

    def __call__(self, planets):
        """Compute gas accretion onto planets

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
    """Pebble accretion model of Bitsch+ (2015) with Bondi regime added.

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
        """Compute the transition mass between Bondi and Hill Regimes.

        args:
            R : radius, AU
            epsilon : approximate power law scaling of pressure with radius

        returns:
            M_t (ndarray): transition mass, Mearth
        """
        h = self._disc.interp(R, self._disc.H) / R
        
        if not epsilon is None:
            eta = 0.5 * h**2 * np.abs(epsilon)
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
        # Replaces Sigma_p -> np.pi * rho_p * r_eff
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
            2 * r_d * Sig_p * delta_v
        )

        return np.array(Mdot)

    def computeMdot(self, Rp, Mp):
        '''
        Calculate the pebble accretion rate.
    
        args:
             Rp : radius of planet in AU
             Mp : mass of planet in M_earth

        returns:
            Mdot (ndarray): Mass accretion rate of pebbles for each planet.
        '''
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
    Planetesimal accretion model based on Danti et al (2023).

    args:
        disc: accretion disc
        gamma: turbulent stirring factor
        rho_p: internal planetesimal/protoplanet density
        eta_ice: iceline location (in AU)
    """
    def __init__(self, disc, gamma=None, rho_p=2, eta_ice=3):
        if gamma is None:
            self._stirring = np.sqrt(disc.alpha)*disc.h
        else:
            self._stirring = gamma*np.ones_like(disc.R)
        self._rho_p = rho_p
        self._eta_ice = eta_ice
        self.set_disc(disc)
        self.dRdt = None

    def set_disc(self, disc):
        self._disc = disc

    def _R_phys(self,Mp):
        """Physical radius of planetesimals"""
        return (3/(4*np.pi*self._rho_p/Msun*AU**3)*Mp*Mearth/Msun)**(1/3)

    def relative_velocity(self, Rp):
        """Calculate planetesimal velocity relative to the gas
        
        return:
            array: combined radial and tangential relative velocity (in AU/code time)
        """
        disc = self._disc
        eta = - np.interp(Rp, reduce(disc.R), np.diff(disc.P) / disc.grid.dRc / reduce(disc.midplane_gas_density)) / disc.star.Omega_k(Rp)
        return np.sqrt((disc.star.v_k(Rp) * eta)**2 + np.interp(Rp,reduce(disc.R),(reduce(disc.v_drift[2]) - disc.gas.viscous_velocity(disc)))**2)
 
    def Reynolds(self, Rp, v = None):
        """Calculate the Reynolds number of planetesimals at given orbital radii"""
        disc = self._disc
        if v is None:
            v = self.relative_velocity(Rp) 
        nu = (disc.visc_mol*Omega0*AU) / (disc.midplane_density*AU**3)
        Re = v * disc.interp(Rp,disc.R_planetesimal / nu)
        return Re
    
    def Mach(self,Rp,v = None):
        """Calculate the Mach number at given orbital radii"""
        if v is None:
            v = self._self.relative_velocity(Rp) 
        c_s = self._disc.cs
        Ma = v / self._disc.interp(Rp,c_s)
        return Ma

    def drag_coeff(self,Rp = None):
        """
        Calculate the drag coefficient given by Podolak et al. (1988).
        """
        vrel = self.relative_velocity(Rp)
        Ma = self.Mach(Rp,vrel)
        Re = self.Reynolds(Rp,vrel)
     
        drag_coeff = np.zeros_like(Ma)

        # Calculate the drag coefficient for the different regimes
        # Apply conditions: Ma < 1 and Re < 10^3
        condition = (Ma < 1) & (Re < 1e3) & (Re > 1)
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
        """Calculate the protoplanet's outer radius.
        
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
        Calculate the protoplanet capture radius 
        according to Valletta & Helled (2021).
        
        args:
            Rp: Protoplanet heliocentric radius (in AU)
            Mp: Protoplanet mass (in Earth masses)
        
        return:
            array: Capture radius (in AU)
        """
        disc    = self._disc
        star    = disc.star

        rH      = star.r_Hill(Rp, Mp*Mearth/Msun)
        D       = self.drag_coeff(Rp)
        R_pla   = disc.R_planetesimal
        rho_p   = self._rho_p

        # Convert Mp to solar masses for calculations
        Mp_solar_masses     = Mp * Mearth / Msun

        # Planet outer radius
        R0      = self.R_p_out(Rp, Mp)

        # Outer density and pressure of planet envelope (equation 4)
        # Interpolate the disc properties and assume value 
        #   to be outermomst envelope value
        P0      = disc.interp(Rp, disc.P)
        rho0    = disc.interp(Rp, disc.midplane_density)

        # Calculate alpha parameter (equation 5) ## Different alpha than viscous
        alpha   = Mp_solar_masses * rho0 / (P0 * R0)
        
        # Calculate rho_star (equation 8) # NOT rho of central star
        rho_star = 2 * R_pla * rho_p / (3 *  D * rH)

        # Calculate capture radius (equation 7)
        R_capt  = R0 / (1 + (1/alpha) * np.log10(rho_star/rho0))
    
        return R_capt
    
    def R_capt(self, Rp, Mp):
        """
        Calculate the protoplanet capture radius.
        args:
            Rp: Protoplanet radius (in AU)
            Mp: Protoplanet mass (in Earth masses)

        return: 
            array: Protoplanet capture radius (in AU)
        """
        # if attached M_Z < M_H-He
        R_captr = self.R_captr_attached(Rp, Mp)
        self._R_captr = R_captr

        return R_captr

    def f_g(self, Rp):
        """Calculate the surface density scaling factor between our disc and the MMSN"""
        # Define standard Sigma
        mmn_ref = 2400 * (Rp)**(-1.5)
        Sigma_G = self._disc.interp(Rp, self._disc.Sigma_G)

        #Return ratio of disc sigma to standard sigma
        fg = Sigma_G / mmn_ref
        return fg

    def inclination(self, Rp):
        """
        Calculate the planetesimal population inclination.
        
        args:
            Rp: Orbital radius (in AU)
        
        returns:
            array: Planetesimal inclination (in radians)
        """
        disc = self._disc

        gamma = disc.interp(Rp,self._stirring) 
        
        fg = self.f_g(Rp)
        
        R_pla = disc.R_planetesimal  # in AU
        rho_p = self._rho_p

        # Calculate edrag using equation 10
        i0 = 0.23 * ((fg) * (gamma**2) * (R_pla*AU/1e5/1.0) * (rho_p/(3.0)))**(1/3) * (Rp/1.0)**(11/12)
    
        return i0

    def computeAccEff(self, Rp, Mp, dRdt):
        """
        Calculate the planetesimal accretion efficiency.

        args:
            Rp: Protoplanet orbital radius (in AU)
            Mp: Protoplanet mass (in Earth masses)
            dRdt: Protoplanet migration rate

        return: 
            array: Planetesimal accretion efficiency
        """
        disc = self._disc
        star = disc.star
        
        rH   = star.r_Hill(Rp, Mp*Mearth/Msun)
        h_p = rH/Rp
        R_captr = self.R_capt(Rp, Mp)
        R_captr /= rH # capture used instead of physical
        
        i0 = self.inclination(Rp) / h_p

        T_k = (2*np.pi) / star.Omega_k(Rp) # Orbital period in 2pi*years

        alpha_pla = 2.5 * np.sqrt(R_captr / (1 + 0.37 * i0*i0 / R_captr))
        beta_pla = 0.79 * (1 + 10 * i0*i0)**(-0.17)

        tau_mig = Rp/np.abs(dRdt) * (h_p**2/T_k)

        b_p = 1 / tau_mig # migration speed

        # Calculate the accretion efficiency
        acc_eff = alpha_pla * b_p ** (beta_pla - 1)
        
        return acc_eff, R_captr

    def computeMdotMigration(self, Rp, Mp, dRdt):
        """
        Compute the planetesimal accretion rate in the case of migration.
        
        args:
            Rp: Protoplanet radius (in AU)
            Mp: Protoplanet mass (in Earth masses)

        return: 
            array: Planetesimal accretion rate (Earth masses/code time)
        """
        disc = self._disc
        Sigma_pla = disc.interp(Rp, disc.Sigma_D[2])
        
        acc_eff = self.computeAccEff(Rp, Mp, dRdt)
        acc_eff_Rp = acc_eff[0]

        # Calculate the planetesimal accretion rate
        Mdot = 2 * np.pi * Rp * (-dRdt) * Sigma_pla * acc_eff_Rp / Mearth * AU**2
        self.dRdt = dRdt
        return Mdot

    def eq_eccentricity_ida2008(self, Rp, r_pltsml = None, iceline = 4):
        """
        Calculate the equilibrium eccentricity of planetesimals based on ida et al (2008).
        This model only uses turbulent stirring
        
        args:
            Rp: Protoplanet location (in AU)
            r_pltsml: Planetesimal radius (AU)
            iceline: Ice line location (AU)

        return:
            array: equilibrium eccentricity from turbulent excitation"""
        disc = self._disc
        if r_pltsml == None:
            r_pltsml = disc.R_planetesimal
    
        eta_ice_arr = np.ones_like(Rp)
        eta_ice_arr[Rp < iceline] *= self._eta_ice # Where iceline is? Assuming 4 AU for now

        # Calculate surface density scaling factors for dust and gas
        Sigma_D_MMSN = 10*eta_ice_arr*Rp**(-3/2)
        f_d = disc.interp(Rp,disc.Sigma_D.sum(0))/Sigma_D_MMSN #planetesimals included?
        f_g = self.f_g(Rp)
        gamma = disc.interp(Rp,self._stirring)
        
        rho_p = self._rho_p

        # Calculate equilibirum eccentricities of turbulent stirring vs tidal damping, drag, and collisional damping
        e_tidal = 24 * f_g**0.5 * gamma * ((r_pltsml*AU/1e5/10**3)**3*rho_p/3)**-0.5 * (Rp)**(3/4)
        e_drag = 0.23 * f_g**(1/3) * gamma**(2/3) * (r_pltsml/(10**5/AU)*rho_p/3)**(1/3) * Rp**(11/12)
        e_coll = 3.6 * f_g * (f_d * eta_ice_arr)**-0.5 * gamma * (r_pltsml/(10**5/AU))**0.5 * (rho_p/3)**(5/6) * Rp**(5/4)
        
        # Return smallest of the three
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
            array: equilibrium eccentricity from planetesimal-planetesimal or protoplanet-planetesimal interactions"""
        disc = self._disc
        m_planetesimal = 4/3*np.pi*(disc.R_planetesimal*AU)**3*disc._rho_s
        
        # eccentricity excited by planetesimal-planetesimal interaction
        em_mm = 20*(m_planetesimal/1e23)**(-1/15)*(Rp)**(9/20)*(2*m_planetesimal/Msun/(3*disc.star.M))**(1/3)
        
        # eccentricity excited by protoplanet-planetesimal interaction
        em_Mm = 6*(m_planetesimal/1e23)**(1/18)*(Rp)**(7/24)*((Mp*Mearth/Msun+m_planetesimal/Msun)/3*disc.star.M)**(1/3)

        return np.max((em_Mm,em_mm),axis=0)

    def compute_v_ran(self, Rp, Mp):
        """
        Calculate the relative velocity between the protoplanet and the planetesimals
        
        args:
            Rp: Protoplanet location (in AU)
            Mp: Protoplanet mass (in Earth masses)

        return: 
            array: Relative velocity (AU/code time)"""
        disc = self._disc
    
        e_eq = self.eq_eccentricity_ida2008(Rp)
        e_eq_2 = self.eq_eccentricity_makino1993(Rp,Mp)
        v_disp = np.max((e_eq,e_eq_2),axis=0) * disc.star.v_k(Rp)

        return v_disp

    def planetesimal_iso_mass(self, Rp):
        """Planetesimal isolation mass when neither planetesimals nor protoplanets are migrating (Earth masses)"""
        return 0.1*(self._disc.interp(Rp,self._disc.Sigma_D[2])/5)**1.5 * (Rp)**3 * (self._disc.star.M)**-0.5

    def computeMdotTwoPhase(self, Rp, Mp, dRdt=None):
        """
        Compute the planetesimal accretion rate in the absence of migration.
        
        args:
            Rp: Protoplanet radius (in AU)
            Mp: Protoplanet mass (in Earth masses)

        return: 
            array: Planetesimal accretion rate (Earth masses/code time unit)
        """
        disc = self._disc
        
        disc._v_drift = np.concat((disc.v_drift,[np.zeros_like(disc.v_drift[1])]))
        
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
        Mdot[filter] = (np.pi*disc.star.Omega_k(Rp_grow)*Sigma_pla/Msun*AU**2*r_physical**2*(1 + v_esc_sqrd/v_rel**2))*Msun/Mearth
        
        return Mdot

    def computeMdot(self, Rp, Mp, dRdt=None):
        """
        Compute the planetesimal accretion rate in migrating and nonmigrating cases.
        
        args:
            Rp: Protoplanet radius (in AU)
            Mp: Protoplanet mass (in Earth masses)
            dRdt: Migration rate (AU/code time)
        """
        Mdot = 0
        if dRdt is None:
            dRdt = np.zeros_like(Rp)
        if (dRdt < 0).any():
            Mdot = self.computeMdotMigration(Rp, Mp, dRdt)
        else:
            Mdot = self.computeMdotTwoPhase(Rp, Mp)

        return Mdot 
    
    def update(self):
        """Update internal quantities after the disc has evolved"""
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
    """Type 1 Migration model of Paardekooper et al (2011)

    Only implemented for sofenting the default sofetning parameter b/h=0.4

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
        M     : central mass, default = 1
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
        nu    = disc.interp(Rp, disc.nu)
        Pr    = disc.interp(Rp, disc.Pr)

        Om_k = star.Omega_k(Rp)
        
        Xi = nu/Pr
        Q = 2*Xi/(3*h*h*h*Rp*Rp*Om_k)
        g_eff = self.gamma_eff_tab(Q)
        
        q_h = (Mp*Mearth/(star.M*Msun)) / h

        jp = Om_k*Rp*Rp
        Om_kr_2 = jp*jp

        # Convert from g cm^-2 AU**4 Omega0**2 to Mearth AU**2 Omega0**2
        norm  = q_h*q_h*Sigma*Om_kr_2 / g_eff
        norm *= AU**2/Mearth
        
        # Compute the scaling factors
        k = jp / (2*np.pi * nu)
        x = (1.1 / g_eff**0.25) * np.sqrt(q_h)

        pnu = 2*np.sqrt(k*x*x*x)/3
        pXi = 3*pnu*np.sqrt(Pr)/2

        Fnu, Gnu, Knu = _F(pnu), _G(pnu), _K(pnu)
        FXi, GXi, KXi = _F(pXi), _G(pXi), _K(pXi)
        
        torque = (_linblad(alpha, beta) +
                  _hs_baro(alpha) * Fnu * Gnu +
                  _cr_baro(alpha) * (1 - Knu) +
                  _hs_entr(alpha, beta, g_eff) * Fnu * FXi * np.sqrt(Gnu*GXi) +
                  _cr_entr(alpha, beta, g_eff) * np.sqrt((1-Knu)*(1-KXi)))


        return norm*torque

    def migration_rate(self, Rp, Mp):
        """Migration rate, dRdt, of the planet"""
        J = Mp*Rp*self._disc.star.v_k(Rp)
        return 2 * (Rp/J) * self.compute_torque(Rp, Mp)
    
    def __call__(self, planets):
        """Migration rate, dRdt, of the planet"""
        return self.migration_rate(planets.R, planets.M)
    

    
class TypeIIMigration(object):
    """Giant planet migration. Uses relation of Baruteau et al (2014)
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
        """Migration rate, dR/dt, of the planet"""
        disc = self._disc
        
        Sigma = disc.interp(Rp, disc.Sigma)
        nu    = disc.interp(Rp, disc.nu)

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
    
class CridaMigration(object):
    """Migration by Type I and Type II with a switch based on the Crida &
    Morbidelli (2007) gap depth criterion.

    args:
        disc  : accretion disc model
        gamma : ratio of specific heats, default=1.4
    """
    def __init__(self, disc, gamma=1.4):
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
        """Compute migration rate"""
        disc = self._disc
        star = disc.star
        
        vr_I  = self._typeI.migration_rate(Rp, Mp)
        vr_II = self._typeII.migration_rate(Rp, Mp)

        Me = Mp*Mearth/Msun
        q = Me / star.M
        rH = star.r_Hill(Rp, Mp)
        nu = disc.interp(Rp, disc.nu)
        H  = disc.interp(Rp, disc.H)

        Re = Rp * star.v_k(Rp) / nu

        P = np.maximum(0.75*H/rH + 50/(q*Re), 0.541)

        fP = np.where(P < 2.4646, 0.25*(P-0.541), 1 - np.exp(-P**0.75/3))

        return fP*vr_I + (1-fP)*vr_II


    def __call__(self, planets):
        """Compute migration rate"""
        return self.migration_rate(planets.R, planets.M)

    def update(self):
        """Update internal quantities after the disc has evolved"""
        self._typeI.update()
        self._typeII.update()
        
    
        
class Bitsch2015Model(object):
    """Pebble accretion + Gas accretion planet formation model based on
    Bisch et al (2015).

    The model is composed of the Hill branch pebble accretion along with
    gas envelope accretion.

    args:
        disc     : accretion disc model
        pb_gas_f : fraction of pebble accretion rate that arrives as gas,
                   default=0.1
        migrate  : Whether to include migration, default=True
        planetesimal_accretion : Whether to include planetesimal accretion, 
                   default=False
        **kwargs : arguments passed to GasAccretion object
    """
    def __init__(self, disc, pb_gas_f=0.1, migrate=True, pebble_accretion=True, gas_accretion=True, planetesimal_accretion=False, **kwargs):

        self._f_gas = pb_gas_f
        self._disc = disc
        
        self._gas_acc = None
        if gas_accretion:
            self._gas_acc = GasAccretion(disc, **kwargs)

        self._peb_acc = None
        if pebble_accretion:
            self._peb_acc = PebbleAccretion(disc)

        self._pl_acc = None
        if disc._planetesimal and planetesimal_accretion:
            self._pl_acc = PlanetesimalAccretion(disc)

        self._migrate = None
        if migrate:
            self._migrate = CridaMigration(disc)

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
        
        if self._pl_acc:
            self._pl_acc.set_disc(disc)

        if self._migrate:
            self._migrate.set_disc(disc)

        self._disc = disc
            
    def update(self):
        """Update internal quantities after the disc has evolved"""
        if self._gas_acc:
            self._gas_acc.update()
        if self._peb_acc:
            self._peb_acc.update()
        if self._pl_acc:
            self._pl_acc.update()
        if self._migrate:
            self._migrate.update()
        
    def insert_new_planet(self, t, R, planets):
        """Set the initial mass of the planets

        args:
            t : current time
            R : AU, formation locations
            planets : planets object to add planets to
        """
        M0 = self._peb_acc.M_transition(R)

        Mc, Me = M0 * (1-self._f_gas), M0*self._f_gas

        if planets.chem:
            Xs, Xg, _ = self._compute_chem(R) #### consider adding planetesimal chemistry to initial planets
        else:
            Xs, Xg = None, None
            
        planets.add_planet(t, R, Mc, Me, Xs, Xg)

    def _compute_chem(self, R_p):
        disc = self._disc
        chem = disc.chem
        
        Xs = []
        Xg = []
        Xs_pla = []

        eps_dust = np.maximum(disc.interp(R_p, disc.dust_frac[:2].sum(0)), 1e-300)

        for spec in chem:
            Xs_i, Xg_i = chem.ice[spec], chem.gas[spec]
            Xs.append(disc.interp(R_p, Xs_i) / eps_dust)
            Xg.append(disc.interp(R_p, Xg_i))

            if self._pl_acc:
                Xs_pla.append(disc.interp(R_p, disc._planetesimal.ice_abund[spec]) / np.maximum(disc.interp(R_p, disc.dust_frac[2]), 1e-300))
            else:
                Xs_pla = np.zeros_like(Xs)

        return np.array(Xs), np.array(Xg), np.array(Xs_pla)

    def integrate(self, dt, planets):
        """Update the planet masses and radii.

        args:
            dt      : Time to integrate for
            planets : Planets container
        """
        if planets.N == 0: return
        self.update()
        
        chem = False
        if planets.chem:
            chem=True

        f = self._f_gas
        def dMdt(R_p, M_core, M_env):
            Mdot_g = np.zeros_like(R_p)
            if self._gas_acc:
                Mdot_g = self._gas_acc.computeMdot(R_p, M_core, M_env)

            Mdot_s = np.zeros_like(R_p)
            if self._peb_acc:
                Mdot_s = self._peb_acc.computeMdot(R_p, M_core + M_env)
                #planets.Mdot += Mdot_s

            return Mdot_s*(1-f), Mdot_g + Mdot_s*f

        def dRdt(R_p, M_core, M_env):
            if self._migrate:
                migration_rate = self._migrate.migration_rate(R_p, M_core + M_env)
                return migration_rate
            else:
                return np.zeros_like(R_p)

        N = planets.N
        Rmin = self._disc.R[0]
        def f_integ(_, y):
            R_p    = y[   :  N]
            M_core = y[N  :2*N]
            M_env  = y[2*N:3*N]

            Rdot = dRdt(R_p, M_core, M_env)

            Mcdot = np.zeros_like(Rdot)
            Medot = np.zeros_like(Rdot)
            
            Mcdot, Medot = dMdt(R_p, M_core, M_env)

            # Compute the mass accretion rate due to planetesimal accretion
            Mdot_pla = np.zeros_like(Mcdot)
            if self._pl_acc:
                Mdot_pla = self._pl_acc.computeMdot(R_p, M_core, Rdot)

            accreted = R_p <= Rmin
            Rdot[accreted] = Mcdot[accreted] = Medot[accreted] = 0
            
            dydt = np.empty_like(y)
            dydt[:N]    = Rdot
            dydt[N:2*N]  = Mcdot + Mdot_pla
            dydt[2*N:3*N] = Medot

            if chem:
                Xs, Xg, Xs_pla =  self._compute_chem(R_p)

                Ms = Mcdot * f / (1-f)
                Mg = np.maximum(Medot - Ms, 0)
                Nspec = Xs.shape[0]

                dydt[ 3       *N:(3+  Nspec)*N] = (Mcdot*Xs + Mdot_pla*Xs_pla).ravel()
                dydt[(3+Nspec)*N:(3+2*Nspec)*N] = (Ms*Xs + Mg*Xg).ravel()
            
            return dydt
            
        integ = ode(f_integ).set_integrator('dopri5', rtol=1e-5, atol=1e-5)

        if chem:
            Chem_core = (planets.M_core * planets.X_core).flat
            Chem_env  = (planets.M_env  * planets.X_env).flat
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env,
                                 Chem_core, Chem_env])
        else:
            X0 = np.concatenate([planets.R, planets.M_core, planets.M_env])
        integ.set_initial_value(X0, 0)
        #print(f"Before integration: R: {planets.R}, M_core: {planets.M_core}, M_env: {planets.M_env}")  # Debugging print
        integ.integrate(dt)
        #print(f"After integration: R: {integ.y[:N]}, M_core: {integ.y[N:2*N]}, M_env: {integ.y[2*N:3*N]}")  # Debugging print
        # Compute the fraction of the core / envelope that was accreted in
        # solids

        planets.R = integ.y[:N]
        planets.M_core = integ.y[N:2*N]
        planets.M_env  = integ.y[2*N:3*N]
        
        if chem:
            Ns = np.prod(planets.X_core.shape)
            Xc = integ.y[3*N   :3*N  +Ns].reshape(-1, N)
            Xe = integ.y[3*N+Ns:3*N+2*Ns].reshape(-1, N)
            planets.X_core = Xc / np.maximum(planets.M_core, 1e-300)
            planets.X_env  = Xe / np.maximum(planets.M_env, 1e-300)           
        

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
    
    
    #######
    # Test the migration rate calculation
    migI  = TypeIMigration(disc)
    migII = TypeIIMigration(disc)

    migCrida = CridaMigration(disc)

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
    #######
    # Test the growth models
    
    # Set up some planet mass / envelope ratios
    #M_p = planets.M
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
        planet_model.insert_new_planet(0, Ri, planets)

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


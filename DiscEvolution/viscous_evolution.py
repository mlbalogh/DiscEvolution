# viscous_evolution.py
#
# Author: R. Booth
# Date: 16 - Nov - 2016
#
# Contains classes for solving the viscous evolution equations.
################################################################################
from __future__ import print_function
import numpy as np
from copy import deepcopy

class ViscousEvolution(object):
    """Solves the 1D viscous evolution equation.

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.
    """

    def __init__(self, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}



    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._X = 2 * np.sqrt(grid.Rc)
        self._dXe = 2 * np.diff(np.sqrt(grid.Re))
        self._dXc = 2 * np.diff(np.sqrt(grid.Rce))
        self._RXdXe = grid.Rc * self._X * self._dXe

    def _init_fluxes(self, disc):
        """Cache the important variables needed for the fluxes"""
        nuX = disc.nu * self._X

        S = np.zeros(len(nuX) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuX

        # Inner boundary
        if self._in_bound == 'Zero':            # Zero torque
            S[0] = 0
        elif self._in_bound == 'Mdot':          # Constant flux (appropriate for power law)
            S[0] = S[1] * self._X[0] / self._X[1]
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary
        if self._bound == 'Zero':               # Zero torque
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2] ** 2 / S[-3]
        elif self._bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S[-1] = S[-2] * self._X[-2] / self._X[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S[-1] = S[-2] * self._X[-1] / self._X[-2]
        else:
            raise ValueError("Error boundary type not recognised")

        self._dS = np.diff(S) / self._dXc

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return 3. * np.diff(self._dS) / self._RXdXe

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return 3. * np.diff(ds) / self._RXdXe

    def viscous_velocity(self, disc, Sigma=None):
        """Compute the radial velocity due to viscosity"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        # Can accept other density specifications (e.g. gas only to match specification of dust drift by Dipierro+18)
        if Sigma is None:
            Sigma = disc.Sigma
               
        RS = Sigma * disc.R
        return np.nan_to_num(- 3 * self._dS[1:-1] / (RS[1:] + RS[:-1])) # Avoid nans when working with Sigma_G -> 0

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        f = self._fluxes()

        # The following makes sure planetesimal 
        # surface density is not evolved with viscous evolution.
        if disc._planetesimal:
            # Update sigma
            Sigma_temp = disc.Sigma + dt * f
        
            Sigma_G_old = disc.Sigma_G
            # Find new dust sigmas
            Sigma_D_new = Sigma_temp*disc.dust_frac[:-1]
            Sigma_G_new = Sigma_G_old*Sigma_temp/disc.Sigma
            Sigma_new = Sigma_G_new + Sigma_D_new.sum(0) + disc.Sigma_D[-1]
            Dust_Frac_New = np.concat((Sigma_D_new / Sigma_new, [disc.Sigma_D[-1] / Sigma_new]),axis=0)
        
            disc._eps = Dust_Frac_New

        else:
            Sigma_new = disc.Sigma + dt * f

        for t in (tracers+adv):
            if t is None: continue
            tracer_density = t*disc.Sigma
            t[:] = (dt*self._tracer_fluxes(t) + tracer_density) / (Sigma_new + 1e-300)

        disc.Sigma[:] = Sigma_new

class ViscousEvolutionFV(object):
    """Solves the 1D viscous evolution equation via a finite-volume method

    This class handles the inclusion of dust species in the one-fluid
    approximation. The total surface density (Sigma = Sigma_G + Sigma_D) is
    updated under the action of viscous forces, which are taken to act only
    on the gas phase species.

    Can optionally update tracer species.

    args:
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.
    """

    def __init__(self, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}'.format(self.__class__.__name__, self._tol)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol)}

    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rh  = np.sqrt(grid.Rc)
        self._dR3 = np.diff(np.sqrt(grid.Rce)**3)

        self._dA = grid.Re
        self._dV = 0.5 * np.diff(grid.Re**2)

    def _init_fluxes(self, disc):
        """Cache the important variables needed for the fluxes"""
        nuRh = disc.nu * self._Rh

        S = np.zeros(len(nuRh) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuRh

        # Inner boundary
        if self._in_bound == 'Zero':            # Zero torque
            S[0] = 0
        elif self._in_bound == 'Mdot':          # Constant flux (appropriate for power law)
            S[0] = S[1] * self._Rh[0] / self._Rh[1]
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary
        if self._bound == 'Zero':
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2] ** 2 / S[-3]
        elif self._bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S[-1] = S[-2] * self._Rh[-2] / self._Rh[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S[-1] = S[-2] * self._Rh[-1] / self._Rh[-2]
        else:
            raise ValueError("Error boundary type not recognised")

        self._dS = 4.5 * np.diff(S) / self._dR3

    def _fluxes(self):
        """Compute the mass fluxes for the viscous evolution equations.

        Gas update from Bath & Pringle (1981)
        """
        return np.diff(self._dA * self._dS) / self._dV

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        # Upwind the tracer density 
        ds = self._dS * np.where(self._dS <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return np.diff(self._dA * ds) / self._dV

    def viscous_velocity(self, disc, S = None):
        """Compute the radial velocity due to viscosity"""
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        if S is None:
            S = disc.Sigma 
        return - 0.5 * self._dS[1:-1] / (S[1:] + S[:-1])

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        tc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()
        return self._tol * tc

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the viscous evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes(disc)

        f = self._fluxes()
        Sigma_new = disc.Sigma + dt * f

        for t in (tracers + adv):
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*f) / (Sigma_new + 1e-300)
        disc.Sigma[:] = Sigma_new


class HybridWindModel(object):
    """Finite volume model for combined viscous evolution + winds
    
    args:
       psi_DW    : Ratio of the viscous alpha to the turbulent one.
       lambda_DW : Wind level arm parameter. Default=3
       tol       : Ratio of the time-step to the maximum stable one.
                   Default = 0.5
       in_bound  : Type of internal boundary condition:
                     'Zero'      : Zero torque boundary
                     'Mdot'      : Constant Mdot (power law).
       boundary  : Type of external boundary condition:
                     'Zero'      : Zero torque boundary
                     'power_law' : Power-law extrapolation
                     'Mdot_inn'  : Constant Mdot, same as inner (power law).
                     'Mdot_out'  : Constant Mdot, for tail of LBP.

    Notes :
       The alpha provided by the disc model is assumed to be the total alpha, i.e.
       the viscous + disk wind alpha.
    """
    def __init__(self, psi_DW, lambda_DW=3, tol=0.5, boundary='power_law', in_bound='Mdot'):
        self._tol = tol
        self._bound = boundary
        self._in_bound = in_bound

        self._lambda = lambda_DW
        self._psi = psi_DW

    def ASCII_header(self):
        """header"""
        return '# {} tol: {}, psi_DW: {}, lambda_DW: {}'.format(self.__class__.__name__, 
            self._tol, self._psi, self._lambda)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, { "tol" : str(self._tol),
                                          "psi_DW" : str(self._psi),
                                          "lambda_DW" : str(self._lambda) }
    def _setup_grid(self, grid):
        """Compute the grid factors"""
        self._Rh  = np.sqrt(grid.Rc)
        self._dR3 = np.diff(np.sqrt(grid.Rce)**3)

        self._dA = grid.Re
        self._dV = 0.5 * np.diff(grid.Re**2)


    def _init_fluxes_visc(self, disc):
        """Compute the flux due to viscosity"""
        nuRh = disc.nu *  self._Rh

        S = np.zeros(len(nuRh) + 2, dtype='f8')
        S[1:-1] = disc.Sigma_G * nuRh

        # Inner boundary
        if self._in_bound == 'Zero':            # Zero torque
            S[0] = 0
        elif self._in_bound == 'Mdot':          # Constant flux (appropriate for power law)
            S[0] = S[1] * self._Rh[0] / self._Rh[1]
        else:
            raise ValueError("Error boundary type not recognised")

        # Outer boundary
        if self._bound == 'Zero':
            S[-1] = 0
        elif self._bound == 'power_law':
            S[-1] = S[-2] ** 2 / S[-3]
        elif self._bound == 'Mdot_out':         # Constant flux (appropriate for tail of LBP)
            S[-1] = S[-2] * self._Rh[-2] / self._Rh[-1]
        elif self._bound == 'Mdot_inn':         # Constant flux (appropriate for power law)
            S[-1] = S[-2] * self._Rh[-1] / self._Rh[-2]
        else:
            raise ValueError("Error boundary type not recognised")

        self._f_visc = 4.5 * np.diff(S) / self._dR3

    def _init_fluxes_wind(self, disc, dt=0):
        """Compute the flux and mass-loss rate due to the wind"""
        #Use first order Donor cell method:
        v_DW = 1.5 * (disc.nu/disc.R) * self._psi

        F = np.zeros(len(disc.Sigma_G) + 1, dtype='f8')
        F[:-1] = v_DW * disc.Sigma_G

        # Outer boundary
        if self._bound == 'Zero':
            F[-1] = 0
        else:
            F[-1] = F[-2] * (v_DW[-1]*disc.R[-1]) / (v_DW[-2]*disc.R[-2])

        # Compute the mass-loss rate:
        loss_rate = v_DW * disc.Sigma_G / (2*(self._lambda-1) * disc.R)


        self._f_wind = F
        self._s_wind = loss_rate

    def viscous_velocity(self, disc, S = None):
        """Compute the radial velocity due to viscosity + winds"""
        self._setup_grid(disc.grid)
        self._init_fluxes_visc(disc)
        self._init_fluxes_wind(disc)

        if S is None:
            S = disc.Sigma 
        return - 0.5 * (self._f_visc + self._f_wind)[1:-1] / (S[1:] + S[:-1])

    def _fluxes(self):
        """Compute the mass fluxes and loss term """
        return np.diff(self._dA * (self._f_visc + self._f_wind)) / self._dV

    def _tracer_fluxes(self, tracers):
        """Compute fluxes of a tracer.

        Uses the viscous update to compute the flux of  Sigma*tracers,
        divide by the updated Sigma to get the new tracer value.
        """
        shape = tracers.shape[:-1] + (tracers.shape[-1] + 2,)
        s = np.zeros(shape, dtype='f8')
        s[..., 1:-1] = tracers
        s[..., 0] = s[..., 1]
        s[..., -1] = s[..., -2]

        F = self._f_visc + self._f_wind

        # Upwind the tracer density 
        ds = F * np.where(F <= 0, s[..., :-1], s[..., 1:])

        # Compute the viscous update
        return np.diff(self._dA * ds) / self._dV

    def max_timestep(self, disc):
        """Courant limited time-step"""
        grid = disc.grid
        nu = disc.nu

        dXe2 = np.diff(2 * np.sqrt(grid.Re)) ** 2

        t_visc = ((dXe2 * grid.Rc) / (2 * 3 * nu)).min()

        v_DW   = 1.5 * (disc.nu/grid.Rc) * self._psi
        t_wind = (np.diff(0.5*grid.Re**2) / (grid.Rc * v_DW)).min()

        return self._tol * min(t_visc, t_wind)

    def __call__(self, dt, disc, tracers=[], adv=[]):
        """Compute one step of the evolution equation
        args:
            dt      : time-step
            disc    : disc we are updating
            tracers : Tracer species to update. Should be a list of arrays with
                      shape = [*, disc.Ncells].
            adv     : Tracers that are advected but not removed due to mass-loss
        """
        self._setup_grid(disc.grid)
        self._init_fluxes_visc(disc)
        self._init_fluxes_wind(disc, dt)

        f = self._fluxes()
        
        # The following makes sure planetesimal 
        # surface density is not evolved with viscous evolution.
        if disc._planetesimal:

            # Update sigma
            Sigma_temp = disc.Sigma + dt * f
        
            Sigma_G_old = disc.Sigma_G
            # Find new dust sigmas
            Sigma_D_new = Sigma_temp*disc.dust_frac[:-1]
            Sigma_G_new = Sigma_G_old*Sigma_temp/disc.Sigma
            Sigma_new = Sigma_G_new + Sigma_D_new.sum(0) + disc.Sigma_D[-1]
            Dust_Frac_New = np.concat((Sigma_D_new / Sigma_new, [disc.Sigma_D[-1] / Sigma_new]),axis=0)
        
            disc._eps = Dust_Frac_New
        else:
            Sigma_new = disc.Sigma + dt * (f - self._s_wind)

        for t in tracers: # Tracers advected + removed
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) - t*f) / (Sigma_new + 1e-300)

        for t in adv: # Tracers advected only
            if t is None: continue
            t[:] += dt*(self._tracer_fluxes(t) +  - (t + self._s_wind)*f) / (Sigma_new + 1e-300)
        
        disc.Sigma[:] = Sigma_new

class LBP_Solution(object):
    """Analytical solution for the evolution of an accretion disc,

    Lynden-Bell & Pringle, (1974).

    args:
        M     : Disc mass
        rc    : Critical radius at t=0
        n_c   : viscosity at rc
        gamma : radial dependence of nu, default=1
    """

    def __init__(self, M, rc, nuc, gamma=1):
        self._rc = rc
        self._nuc = nuc
        self._tc = rc * rc / (3 * (2 - gamma) ** 2 * nuc)

        self._Sigma0 = M * (2 - gamma) / (2 * np.pi * rc ** 2)
        self._gamma = gamma

    def __call__(self, R, t):
        """Surface density at R and t"""
        tt = t / self._tc + 1
        X = R / self._rc
        Xg = X ** - self._gamma
        ft = tt ** ((-2.5 + self._gamma) / (2 - self._gamma))

        return self._Sigma0 * ft * Xg * np.exp(- Xg * X * X / tt)

class TaboneSolution(object):
    """Analytical solution for the evolution of a hybrid viscous and
    wind-driven accretion disc,

    Tabone et al. (2022)

    args:
        M      : Disc mass
        rc     : Critical radius at t=0
        n_c    : viscosity at rc (from alpha_ss)
        psi_DW : Ratio of viscous to wind-driven alpha
        d2g    : Dust to gas ratio
        lambda_dW : mass-loss efficiency
    """

    def __init__(self, M, rc, nuc, psi_DW, d2g = 0.01, lambda_DW=3):
        from scipy.special import gamma as gamma_fun
        self._rc = rc
        self._nuc = nuc
        self._tc = rc * rc / (3 * nuc * (1 + psi_DW))

        self._psi = psi_DW
        
        C = 4*psi_DW/((lambda_DW-1)*(1+psi_DW)**2)
        self._xi = 0.25*(1 + psi_DW) * (np.sqrt(1 + C)-1) 
        self._Sigma0 = M * (1-d2g) / (2 * np.pi * rc ** 2 * gamma_fun(1 + self._xi))

    def __call__(self, R, t):
        """Surface density at R and t"""
        tt = t / ((1 + self._psi)*self._tc) + 1
        X = R / (self._rc*tt)
        Xg = X ** -(1-self._xi)
        ft = tt ** -(2.5 + self._xi + self._psi/2)

        return self._Sigma0 * ft * Xg * np.exp(- X)
    
    def viscous_velocity(self, disc, Sigma = None):
        """Compute the radial velocity of gas due to viscosity + winds"""
        if Sigma is None:
            Sigma = disc.Sigma
        v_dw = -1.5*self._psi * disc.nu/disc.R
        
        v_vs = -4.5 / ((disc.Sigma[:-1]+disc.Sigma[1:]) ) * np.diff(disc.nu * disc.Sigma * disc.R**0.5)/np.diff(disc.grid.Rc**1.5)
        
        return ((v_dw[:-1] + v_dw[1:])/2 + v_vs)/2

    def ASCII_header(self):
        """header"""
        return '# {}'.format(self.__class__.__name__)

    def HDF5_attributes(self):
        """Class information for HDF5 headers"""
        return self.__class__.__name__, ""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from disc import AccretionDisc
    from grid import Grid
    from constants import AU, Msun
    from eos import LocallyIsothermalEOS
    from star import SimpleStar

    alpha = 5e-3

    M = 1e-2 * Msun
    Rd = 30.
    T0 = (2 * np.pi)

    grid = Grid(0.1, 1000, 1000, spacing='natural')
    star = SimpleStar()
    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)

    nud = np.interp(Rd, grid.Rc, eos.nu)
    sol = LBP_Solution(M, Rd, nud, 1)

    Sigma = sol(grid.Rc, 0.)

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = ViscousEvolutionFV()

    plt.figure()

    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6]) * T0

    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        l, = plt.loglog(grid.Rc, disc.Sigma / AU ** 2)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t) / AU ** 2,
                        c=l.get_color(), ls='--')

    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma\,[\mathrm{g\,cm}]^{-2}$')

    sol = TaboneSolution(M / AU**2, Rd, nud, psi_DW=1)

    Sigma = sol(grid.Rc, 0.)
    fig1, ax1 = plt.subplots()
    for t in times:
        ax1.loglog(grid.Rc,sol(grid.Rc,t))
    ax1.set_title("Tabone")
    plt.show()

    disc = AccretionDisc(grid, star, eos, Sigma)

    visc = HybridWindModel(psi_DW=1)

    plt.figure()

    # Integrate to given times
    times = np.array([0, 1e4, 1e5, 1e6, 3e6]) * T0
    t = 0
    n = 0
    for ti in times:
        while t < ti:
            dt = visc.max_timestep(disc)
            dti = min(dt, ti - t)

            visc(dti, disc)

            t = min(t + dt, ti)
            n += 1

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} yr'.format(t / (2 * np.pi)))
                print('dt: {} yr'.format(dt / (2 * np.pi)))

        l, = plt.loglog(grid.Rc, disc.Sigma)
        l, = plt.loglog(grid.Rc, sol(grid.Rc, t),
                        c=l.get_color(), ls='--')

    plt.xlabel('$R\,[\mathrm{au}]$')
    plt.ylabel('$\Sigma\,[\mathrm{g\,cm}]^{-2}$')
    plt.show()


    plt.show()

import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('/Users/yuvan/GitHub/DiscEvolution/')

import numpy as np
import matplotlib.pyplot as plt

from DiscEvolution.constants import *
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import *
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import *
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund, SimpleCNOAtomAbund, SimpleCNOMolAbund,
    SimpleCNOChemOberg, TimeDepCNOChemOberg, TimeDepCOChemOberg, EquilibriumCOChemMadhu,
    EquilibriumCNOChemOberg, SimpleCOAtomAbund, SimpleCOChemOberg,
    SimpleCNOChemMadhu, EquilibriumCNOChemMadhu, EquilibriumCOChemOberg
)

import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()
plt.rcParams.update({'font.size': 16})

#### specify separate temperature 
### initial accreation rate: 2pi r sigma radial velocity at bin 0.

class ChemoDynamicsModel(object):
    """
    """

    def __init__(self, disc, chem=None,
                 diffusion=False, radial_drift=False, viscous_evo=False,
                 Sc=1, t0=0):

        self._disc = disc
        self._chem = chem

        self._visc = None
        if viscous_evo:
            self._visc = ViscousEvolution()

        self._diffusion = None
        if diffusion:
            diffusion = TracerDiffusion(Sc)

        # Diffusion can be handled by the radial drift object, without dust we
        # include it ourself.
        self._radial_drift = None
        if radial_drift:
            self._radial_drift = SingleFluidDrift(diffusion)
        else:
            self._diffusion = diffusion

        self._t = t0

    def __call__(self, tmax):
        """Evolve the disc for a single timestep

        args:
            dtmax : Upper limit to time-step

        returns:
            dt : Time step taken
        """
        # Compute the maximum time-step
        dt = tmax - self.t
        if self._visc:
            dt = min(dt, self._visc.max_timestep(self._disc))
        if self._radial_drift:
            dt = min(dt, self._radial_drift.max_timestep(self._disc))

        disc = self._disc

        gas_chem, ice_chem = None, None
        try:
            gas_chem = disc.chem.gas.data
            ice_chem = disc.chem.ice.data
        except AttributeError:
            pass

        # Do Advection-diffusion update
        if self._visc:
            dust = None
            try:
                dust = disc.dust_frac
            except AttributeError:
                pass
            self._visc(dt, disc, [dust, gas_chem, ice_chem])

        if self._radial_drift:
            self._radial_drift(dt, disc,
                               gas_tracers=gas_chem,
                               dust_tracers=ice_chem)

        if self._diffusion:
            if gas_chem is not None:
                gas_chem[:] += dt * self._diffusion(disc, gas_chem)
            if ice_chem is not None:
                ice_chem[:] += dt * self._diffusion(disc, ice_chem)

        # Pin the values to >= 0:
        disc.Sigma[:] = np.maximum(disc.Sigma, 0)
        disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
        if self._chem:
            disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
            disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

        # Chemistry
        if self._chem:
            rho = disc.midplane_gas_density
            eps = disc.dust_frac.sum(0)
            T = disc.T

            self._chem.update(dt, T, rho, eps, disc.chem)

            # If we have dust, we should update it now the ice fraction has
            # changed
            disc.update_ices(disc.chem.ice)

        # Now we should update the auxillary properties, do grain growth etc
        disc.update(dt)

        self._t += dt
        return dt

    @property
    def disc(self):
        return self._disc

    @property
    def t(self):
        return self._t

    def dump(self, filename):
        """Write the current state to a file, including header information"""

        # Put together a header containing information about the physics
        # included
        head = self.disc.header() + '\n'
        if self._visc:
            head += self._visc.header() + '\n'
        if self._radial_drift:
            head += self._radial_drift.header() + '\n'
        if self._diffusion:
            head += self._diffusion.header() + '\n'
        if self._chem:
            head += self._chem.header() + '\n'

        with open(filename, 'w') as f:
            f.write(head, '# time: {}yr\n'.format(self.t / (2 * np.pi)))

            # Construct the list of variables that we are going to print
            Ncell = self.disc.Ncells

            Ndust = 0
            try:
                Ndust = self.disc.dust_frac.shape[0]
            except AttributeError:
                pass

            head = '# R Sigma T'
            for i in range(Ndust):
                head += ' epsilon[{}]'.format(i)
            for i in range(Ndust):
                head += ' a[{}]'.format(i)
            chem = None
            try:
                chem = self.disc.chem
                for k in chem.gas:
                    head += ' {}'.format(k)
                for k in chem.ice:
                    head += ' s{}'.format(k)
            except AttributeError:
                pass

            f.write(head+'\n')

            R, Sig, T = self.disc.R, self.disc.Sigma, self.disc.T
            for i in range(Ncell):
                f.write('{} {} {}'.format(R[i], Sig[i], T[i]))
                for j in range(Ndust):
                    f.write(' {}'.format(self.disc.dust_frac[j, i]))
                for j in range(Ndust):
                    f.write(' {}',format(self.disc.grain_size[j, i]))
                if chem:
                    for k in chem.gas:
                        f.write(' {}'.format(chem.gas[k][i]))
                    for k in chem.ice:
                        f.write(' {}'.format(chem.ice[k][i]))
                f.write('\n')

def run_model(config):
    """
    Run the disk evolution model and plot the results.
    
    Parameters:
    config (dict): Configuration dictionary containing all parameters.
    """
    # Extract parameters from config
    grid_params = config['grid']
    sim_params = config['simulation']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planet']
    chemistry_params = config["chemistry"]
    
    # Set up disc
    # ========================
    # Create the grid
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], grid_params['spacing'])
    

    # Create the star
    star = SimpleStar(M=1, R=2.5, T_eff=4000.)

    # Create time array
    if sim_params['t_interval'] == "power":
        # Determine the number of points needed
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            times = np.logspace(0, np.log10(sim_params['t_final']), num=num_points) * 2 * np.pi
        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            times = np.logspace(np.log10(sim_params['t_initial']), np.log10(sim_params['t_final']), num=num_points) * 2 * np.pi
    elif type(sim_params['t_interval']) == list:
        times = np.array(sim_params['t_interval']) * 2 * np.pi * 1e6
    else:
        times = np.arange(sim_params['t_initial'], sim_params['t_final'], sim_params['t_interval']) * 2 * np.pi

    Mdot = disc_params['Mdot']
    alpha = disc_params['alpha']
    Rd = disc_params['Rd']
    chem_type = chemistry_params['chem_model']

    R_in = 0.5
    R_out = 500

    N_cell = 1000

    T0 = 2 * np.pi

    Mdot *= Msun / (2 * np.pi)
    Mdot /= AU ** 2

    eos_type = 'irradiated'
    # eos_type = 'isothermal'

    # Gas fraction for pebble accretion
    pb_gas_f = 0.0

    output = False
    planets = False
    plot = True
    injection_times = np.arange(0, 3.01e6, 1e5) * T0
    injection_radii = np.logspace(0.5, 2, 16)

    # Initialize the disc model
    grid = Grid(R_in, R_out, N_cell, spacing='natural')
    star = SimpleStar(M=1, R=2.5, T_eff=4000.)

    R = grid.Rc

    eos = LocallyIsothermalEOS(star, 1 / 30., -0.25, alpha)
    eos.set_grid(grid)
    Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)
    if eos_type != 'isothermal':
        # Use a non accreting model to guess the initial density
        eos = IrradiatedEOS(star, alpha, tol=1e-3, accrete=False)
        eos.set_grid(grid)
        eos.update(0, Sigma)

        # Do a new guess for the surface density and initial eos.
        Sigma = (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd)

        eos = IrradiatedEOS(star, alpha, tol=1e-3)
        eos.set_grid(grid)
        # Iterate to constant Mdot
        for i in range(100):
            eos.update(0, Sigma)
            Sigma = 0.5 * (Sigma +
                           (Mdot / (3 * np.pi * eos.nu)) * np.exp(-grid.Rc / Rd))
        eos.update(0, Sigma)

    # Initialize the complete disc object
    disc = DustGrowthTwoPop(grid, star, eos, 0.01, Sigma=Sigma, feedback=True)

    # Initialize the chemistry
    if chem_type == 'TimeDep':
        chemical_model = TimeDepCOChemOberg(a=1e-5)
    elif chem_type == 'Madhu':
        chemical_model = EquilibriumCOChemMadhu(fix_ratios=False, a=1e-5)
    elif chem_type == 'Oberg':
        chemical_model = EquilibriumCOChemOberg(fix_ratios=False, a=1e-5)
    elif chem_type == 'NoReact':
        chemical_model = EquilibriumCOChemOberg(fix_ratios=True, a=1e-5)

    # Initial abundances:
    X_solar = SimpleCOAtomAbund(N_cell)
    X_solar.set_solar_abundances()

    # Iterate as the ice fraction changes the dust-to-gas ratio
    for i in range(10):
        chem = chemical_model.equilibrium_chem(disc.T,
                                               disc.midplane_gas_density,
                                               disc.dust_frac.sum(0),
                                               X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)
    disc.chem = chem

    disc.update_ices(disc.chem.ice)

    # Setup the chemo-dynamical model
    evo = ChemoDynamicsModel(disc, chem=chemical_model,
                             viscous_evo=True,
                             radial_drift=True,
                             diffusion=True)

    # Run model
    # ========================
    t = 0
    n = 0
    
    # Prepare plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Set up initial plot lines
    try:
        axes[0].plot(grid.Rc, 0*grid.Rc, '-', color='black', label="Gas")
        axes[0].plot(grid.Rc, 0*grid.Rc, linestyle="dashed", color='black', label="Dust")
        #axes[0].plot(grid.Rc, 0*grid.Rc, linestyle="dotted", color='black', label="Small dust")
        #axes[0].plot(grid.Rc, 0*grid.Rc, linestyle="dashed", color='black', label="Large dust")
    except:
        pass
    
    # Figure for planet growth track and planetesimal capture radius
    if planet_params['include_planets']:
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
        planet_growth_track = []
        planet_capture_radius = []

    # this is to synchronize colors
    d = 0 
    colors = ["black", "red", "green", "blue", "cyan"]

    end = False

    for ti in times:
        while evo.t < ti and end ==False:
            dt = evo(ti)
            
            t += dt
            n += 1

            if (n % 1000) == 0:
                print('\rNstep: {}'.format(n), end="", flush="True")
                print('\rTime: {} yr'.format(t / (2 * np.pi)), end="", flush="True")
                print('\rdt: {} yr'.format(dt / (2 * np.pi)), end="", flush="True")

        #if False:
        #    try:
        #        l, = axes[0].loglog(grid.Rc, disc.Sigma_G, label='t = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 2)))
        #        axes[0].set_xlabel('$R\\,[\\mathrm{au}]$')
        #        axes[0].set_ylabel('$\\Sigma_{\\mathrm{Gas}} [g/cm^2]$')
        #        #axes[0].set_ylim(ymin=1e-6)
        #        axes[0].set_title('Gas Surface Density')
        #        axes[0].legend()
        #    except:
        #        axes.loglog(grid.Rc, disc.Sigma_G, label='t = {} yrs'.format(np.round(t / (2 * np.pi))))
        #        axes.set_xlabel('$R\\,[\\mathrm{au}]$')
        #        axes.set_ylabel('$\\Sigma_{\\mathrm{Gas}} [g/cm^2]$')
        #        #axes.set_ylim(ymin=1e-6, ymax=1e6)
        #        axes.set_title('Gas Surface Density')
        #        axes.legend()

        if transport_params['radial_drift']:

            l, = axes[0].loglog(grid.Rc, disc.dust_frac.sum(0), linestyle="dashed", label='t = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 2)), color=colors[d])
            axes[0].set_xlabel('$R\\,[\\mathrm{au}]$')
            #axes[1].set_ylabel('$\\Sigma_{\\mathrm{Dust}} [g/cm^2]$')
            axes[0].set_ylabel('$\epsilon$')
            axes[0].set_title('Dust Fraction')
            axes[0].set_xlim(0.7, 300)
            axes[0].set_ylim(10**(-6), 10**(0))
            #axes[1].loglog(grid.Rc, (disc.Sigma_D[0]+disc.Sigma_D[1]), linestyle="dashed", color=l.get_color())

            d+=1

            axes[1].loglog(grid.Rc, disc.grain_size[1], linestyle="dashed", color=l.get_color())
            axes[1].set_xlabel('$R\\,[\\mathrm{au}]$')
            #axes[1].set_ylabel('$\\Sigma_{\\mathrm{Dust}} [g/cm^2]$')
            axes[1].set_ylabel('$a [cm]$')
            axes[1].set_title('Grain Size')
            axes[1].set_xlim(0.7, 300)
            axes[1].set_ylim(10**(-5), 10**(2))

        #if transport_params['radial_drift']:
            #axes[2].plot(grid.Rc, 0*grid.Rc, label='t = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 2)), color=l.get_color())
            #axes[2].loglog(grid.Rc, disc.v_drift[0], linestyle="dotted", color=l.get_color())
            #axes[2].loglog(grid.Rc, disc.v_drift[1], linestyle='dashed', color=l.get_color())

            #axes[2].set_xlabel('$R\\,[\\mathrm{au}]$')
            #axes[2].set_ylabel('Drift Velocity')
            #axes[2].set_ylim(ymin=1e-6)
            #axes[2].set_title('Drift Velocity')
            #axes[2].legend()
            
        if chemistry_params["on"]:
            atom_abund = disc.chem.gas.atomic_abundance()

            #l, = plt.semilogx(R, chem.gas['H2O']/mol_solar['H2O'], '-')
            #plt.semilogx(R, chem.ice['H2O']/mol_solar['H2O'],'--', c=l.get_color())
            plt.semilogx(R, atom_abund.number_abund("C")/atom_abund.number_abund("O"), label=f"{t/10**6:2f} Myr", linestyle="dashed", color=l.get_color())
            plt.xlim(0.7, 300)
            plt.ylim(0, 1.2)
            plt.ylabel('[C/O]')
            plt.legend()

            axes[2].semilogx(R, atom_abund.number_abund('C')/X_solar.number_abund("C"), label=f"{t/10**6:2f} Myr", linestyle="dashed", color=l.get_color())
            # atom_abund.number_abund('C')/X_solar.number_abund("C")
            axes[2].set_ylabel("$[C/H]_{solar}$") 
            axes[2].set_xlim(0.7, 300)
            #axes[2].set_ylim(0, 6)
            #axes[1].set_ylim(ymin=1e-6, ymax=1e6)
            #plt.semilogx(R, chem.gas['CO2'], label=f"CO2 gas: t={t:2e}")



    #plt.tight_layout()
    #plt.xlabel('Radius (AU)')

    #plt.legend(bbox_to_anchor=[1, 1])

    #plt.savefig('graphs/fixed?_H2O_abund_over_t.png', bbox_inches='tight')
    print("outputting graphs")
    fig.savefig('graphs/T&E/test_3_chemo.png', bbox_inches='tight')

# Example usage:
# With planetesimals
#run_model(config)
#print("--- %s seconds ---" % (time.time() - start_time))
# Without planetesimals
# config['planetesimal']['include_planetesimals'] = False
# run_model(config)

# # Without planetesimals and dust
# config['planetesimal']['include_planetesimals'] = False
# config['disc']['d2g'] = 0
# config['transport']['radial_drift'] = False
# run_model(config)

if __name__ == "__main__":
    config = {
        "grid": {
            "rmin": 0.5,
            "rmax": 500,
            "nr": 1000,
            "spacing": "natural"
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1e6,
            "t_interval": [0, 0.1, 0.5, 1, 3], # Myr
        },
        "disc": {
            "alpha": 1e-3,
            "M": 0.05,
            "Rc": 35,
            "d2g": 0.01,
            "Mdot": 1e-8,
            "Sc": 1.0, # schmidt number
            "Rd": 100.
        },
        "eos": {
            "type": "LocallyIsothermalEOS",
            "h0": 1/30,
            "q": -0.25
        },
        "transport": {
            "gas_transport": True,
            "radial_drift": True,
            "diffusion": True, 
            "van_leer": True
        },
        "dust_growth": {
            "feedback": True,
            "settling": False,
            "f_ice": 0.9,           # Set ice fraction to 0
            "uf_0": 100, #500          # Fragmentation velocity for ice-free grains (cm/s)
            "uf_ice": 1000,        # Set same as uf_0 to ignore ice effects
            "thresh": 0.1        # Set high threshold to prevent ice effects
        },
        "chemistry": {
            "on"   : True, 
            "type" : "NoReact", 
            "fix_mu" : True,
            "mu"     : 2.4,
            "crate" : 1e-17,
            "use_abundance_file" : True,
            "abundances" : "Eistrup2016.dat",
            "normalize" : True,
            "variable_grain_size" : True,
            "chem_model": "Oberg"
        },
        "planet": {
            "include_planets": False,
            "Rp": [3],    # initial position of embryo [AU]
            "Mp": [1]     # initial mass of embryo [M_Earth]
        }
    }
    run_model(config)
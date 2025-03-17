import os 
import sys

sys.path.append(os.path.abspath(os.path.join('..')) + '/DiscEvolution')

import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import Msun, AU, yr
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar, MesaStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import DustGrowthTwoPop, PlanetesimalFormation, FixedSizeDust
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolutionFV
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import PlanetDiscDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.planet_formation import *

from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund, SimpleCNOAtomAbund, SimpleCNOMolAbund,
    SimpleCNOChemOberg, TimeDepCNOChemOberg,
    EquilibriumCNOChemOberg,
    SimpleCNOChemMadhu, EquilibriumCNOChemMadhu
)

try:
    from DiscEvolution.chemistry.krome_chem import (
        KromeIceAbund, KromeGasAbund, KromeMolecularIceAbund, KromeChem, 
        UserDust2GasCallBack)
except ImportError:
    # UserDust2GasCallBack must have a definition for this file to compile,
    # but this will never be used if krome_chem is not used
    UserDust2GasCallBack = object


from DiscEvolution.photoevaporation import (
    FixedExternalEvaporation, TimeExternalEvaporation)
from DiscEvolution.internal_photo import ConstantInternalPhotoevap

###############################################################################
# Global Constants
###############################################################################
DefaultModel = "planet_formation_scripts/LenzConfig.json"

###############################################################################
# Global Functions
###############################################################################

# Chemistry Models
# =============================================================================
class KromeCallBack(UserDust2GasCallBack):
    """Call back function for KROME user routines.

    This class does two things:
        1) Sets a globally constant cosmic ray ionization rate
        2) Sets the dust-to-gas ratio at each time step
        
    args:
        cosmic_ray_rate : float, default=1e-17
            Ionization rate of H_2 due to cosmic rays (per second)
    """
    def __init__(self, cosmic_ray_rate=1e-17, grain_size=1e-5):
        super(KromeCallBack, self).__init__(grain_size)
        self._crate = cosmic_ray_rate
    def init_krome(self, krome):
        super(KromeCallBack, self).init_krome(krome)
        try:
            krome.lib.krome_set_user_crate(self._crate)
        except AttributeError:
            pass

class VariableAmax_KromeCallBack(KromeCallBack):
    def __init__(self, cosmic_ray_rate=1e-17, grain_size=1e-5):
        super(VariableAmax_KromeCallBack, self).__init__(cosmic_ray_rate, 
                                                         grain_size)

    def __call__(self, krome, T, rho, dust_frac, **kwargs):
        # Set grain size
        if 'grain_size' in kwargs:
            asize = (kwargs['grain_size']*self._asize)**0.5
            krome.lib.krome_set_user_asize(asize)

        # Set other parameters
        super(VariableAmax_KromeCallBack, self).__call__(krome, T, rho, 
                                                         dust_frac, **kwargs)

def init_abundances_from_file(model, abund, disc):

    gas = abund.gas
    ice = abund.ice

    init_abund = np.genfromtxt(model['chemistry']['abundances'],
                               names=True, dtype=('|S5', 'f8', 'f8'),
                               skip_header=1)

    for name, value in zip(init_abund['Species'], init_abund['Abundance']):
        name = name.decode()
        if name in ice.species:
            value += abund.ice.number_abund(name)
            abund.ice.set_number_abund(name,value)
        elif name in gas.species:
            value += abund.gas.number_abund(name)
            abund.gas.set_number_abund(name, value)
        else:
            pass

    return abund

def setup_init_abund_krome(model, disc):
    Ncell = model['grid']['N']

    gas = KromeGasAbund(Ncell)
    ice = KromeIceAbund(Ncell)
        
    abund = KromeMolecularIceAbund(gas,ice)

    abund.gas.data[:] = 0
    abund.ice.data[:] = 0

    abund = init_abundances_from_file(model, abund, disc)

    if model['chemistry']['normalize']:
        norm = 1 / (abund.gas.total_abund + abund.ice.total_abund)
        abund.gas.data[:] *= norm
        abund.ice.data[:] *= norm

    # Add dust
    abund.gas.data[:] *= (1-disc.dust_frac.sum(0))
    abund.ice.data[:] *= (1-disc.dust_frac.sum(0))
    abund.ice["grain"] = disc.dust_frac.sum(0)

    return abund

def get_simple_chemistry_model(model):
    chem_type = model['chemistry']['type']

    grain_size = 1e-5
    try:
        grain_size = model['chemistry']['fixed_grain_size']
    except KeyError:
        pass
    
    if chem_type == 'TimeDep':
        chemistry = TimeDepCNOChemOberg(a=grain_size)
    elif chem_type == 'Madhu':
        chemistry = EquilibriumCNOChemMadhu(fix_ratios=False, a=grain_size)
    elif chem_type == 'Oberg':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=False, a=grain_size)
    elif chem_type == 'NoReact':
        chemistry = EquilibriumCNOChemOberg(fix_ratios=True, a=grain_size)
    else:
        raise ValueError("Unkown chemical model type")

    return chemistry
   
def setup_init_abund_simple(model, disc):
    chemistry = get_simple_chemistry_model(model)

    X_solar = SimpleCNOAtomAbund(model['grid']['N'])
    X_solar.set_solar_abundances()

    # Iterate as the ice fraction changes the dust-to-gas ratio
    for i in range(10):
        chem = chemistry.equilibrium_chem(disc.T,
                                          disc.midplane_gas_density,
                                          disc.dust_frac.sum(0),
                                          X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)

    # If we have abundances from file, overwrite the previous calculation:
    if model['chemistry'].get('use_abundance_file', False):
        for s in chem.gas.names:
            if 'grain' not in s:
                chem.gas.set_number_abund(s, 0.)
                chem.ice.set_number_abund(s, 0.)
        
        chem = init_abundances_from_file(model, chem, disc)
        disc.initialize_dust_density(chem.ice.total_abund)


    return chem

def setup_krome_chem(model):
    if model['chemistry']['fix_mu']:
        mu = model['chemistry']['mu']
    else:
        mu = 0.

    crate = 1e-17
    try:
        crate = model['chemistry']['crate']
    except KeyError:
        pass
    grain_size = 1e-5
    try:
        grain_size = model['chemistry']['fixed_grain_size']
    except KeyError:
        pass
    
    call_back = KromeCallBack(crate, grain_size)
    try:
        if model['chemistry']['variable_grain_size']:
            call_back = VariableAmax_KromeCallBack(crate, grain_size)
    except KeyError:
        pass

    chemistry = KromeChem(renormalize=model['chemistry']['normalize'],
                          fixed_mu=mu, call_back=call_back)

    return chemistry

def setup_simple_chem(model):
    return get_simple_chemistry_model(model)


# Disk Models
# =============================================================================
def LBP_model(**kwargs):
    init_params = {"alpha":1e-3, "rmin":0.05, "rmax":1000, "nr":1000, "Mdot":1e-8, "fixed": False}
    init_params.update(kwargs)

    alpha = init_params['alpha']

    grid = Grid(init_params['rmin'], init_params['rmax'], init_params['nr'])
    star = SimpleStar()

    Mdot  = init_params["Mdot"]
    Rd    = 30
    R = grid.Rc
    Mdot *= (Msun / yr) / AU**2
    Sigma = (Mdot / (0.1 * alpha * R**2 * star.Omega_k(R))) * np.exp(-R/Rd)

    eos = SimpleDiscEOS(star, alpha)    
    eos.set_grid(grid)
    eos.update(0, Sigma)
    for i in range(100):
        Sigma = 0.5 * (Sigma + (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd))
        eos.update(0, Sigma)

    gas = ViscousEvolutionFV()

    disc = DustGrowthTwoPop(grid, star, eos, 0.01, Sigma)

    if init_params['fixed'] == True:
        disc = FixedSizeDust(grid, star, eos, 0.01, 1, Sigma)
    v_visc = gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(v_visc)[0]
    Mtot = disc.Mtot()/Msun

    return disc, gas

def setup_planets(model, disc):
    if 'planets' not in model:
        return None

    planet_params = model['planets']
        
    planets = Planets(Nchem = 0)
    
    Rp = planet_params['Rp']
    Mp = planet_params['Mp']

    for i in range(len(Rp)):
        planets.add_planet(0, Rp[i], Mp[i], 0)

    if planet_params['planet model'] == 'Bitsch2015Model':
        planet_model = Bitsch2015Model(disc, pb_gas_f=0.1)

    return planets, planet_model
    

def setup_model(model, start_time):
    '''Setup the physics of the model'''
    
    gas           = None
    dust          = None
    diffuse       = None
    planetesimal  = None
    chemistry     = None
    ext_photoevap = None
    int_photoevap = None

    grid_model = model['grid']
    disc_model = model['disc']
    disc, gas = LBP_model(alpha=disc_model['alpha'], rmin=grid_model['R0'], rmax=grid_model['R1'], 
                                                  nr=grid_model['N'], Mdot=grid_model['Mdot'], fixed=False)

    # Setup the chemical part of the disc
    if model['chemistry']["on"]:
        if model['chemistry']['type'] == 'krome':
            disc.chem = setup_init_abund_krome(model)
            disc.update_ices(disc.chem.ice)
        else:
            disc.chem =  setup_init_abund_simple(model, disc)
            disc.update_ices(disc.chem.ice)

    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])

    if model['transport']['radial drift']:
        van_leer = model['dust_transport']['van leer']
        settling = model['dust_transport']['settling']
        
        if model['dust_transport']['diffusion']:
            dust_diffusion = diffuse
            diffuse = None
        else:
            dust_diffusion = None

        dust = SingleFluidDrift(diffusion=dust_diffusion, 
                                settling=settling,
                                van_leer=van_leer)
        
    if model['photoevaporation']['on']:
        if model['photoevaporation']['method'] == 'const':
            ext_photoevap = \
                FixedExternalEvaporation(model['photoevaporation']['coeff'])
        elif model['photoevaporation']['method'] == 'internal_const':
            int_photoevap = \
                ConstantInternalPhotoevap(model['photoevaporation']['coeff'])
        else:
            raise ValueError("Photoevaporation method not present in run_model")

    if model['chemistry']['on']:
        if  model['chemistry']['type'] == 'krome':
            chemistry = setup_krome_chem(model)
        else:
            chemistry = setup_simple_chem(model)

    if model['planetesimal']['active']:
        disc._planetesimal = True
        planetesimal_params = model['planetesimal']
        planetesimal = PlanetesimalFormation(disc, d_planetesimal=planetesimal_params['diameter'], St_min=planetesimal_params['St_min'], 
                                             St_max=planetesimal_params['St_max'], pla_eff=planetesimal_params['pla_eff'])

    planets, planet_model = setup_planets(model, disc)

    return PlanetDiscDriver(disc, gas=gas, dust=dust, planetesimal=planetesimal, diffusion=diffuse, planets=planets, planet_model=planet_model, t0=start_time)


def setup_output(model):
    
    out = model['output']

    # Setup of the output controller
    output_times = np.arange(out['first'], out['last'], out['interval'])
    if not np.allclose(out['last'], output_times[-1], 1e-12):
        output_times = np.append(output_times, out['last'])

    output_times *= yr

    if out['plot']:
        plot = np.array(out["plot_times"]) * yr
    else:
        plot = []

    EC = Event_Controller(save=output_times, plot=plot)
    
    # Base string for output:
    mkdir_p(out['directory'])
    base_name = os.path.join(out['directory'], out['base'] + '_{:04d}')

    format = out['format']
    if format.lower() == 'hdf5':
        base_name += '.h5'
    elif format.lower() == 'ascii':
        base_name += '.dat'
    else:
        raise ValueError ("Output format {} not recognized".format(format))

    return base_name, EC

def _plot_grid(model, figs=None):

    if figs is None:
        try:
            model.disc.dust_frac
            f, subs = plt.subplots(2,2)
        except AttributeError:
            f, subs = plt.subplots(1,1)
            subs = [[subs]]
    else:
        f, subs = figs
        
    grid = model.disc.grid
    try:
        eps = model.disc.dust_frac.sum(0)

        subs[0][1].loglog(grid.Rc, eps)
        subs[0][1].set_xlabel('$R$')
        subs[0][1].set_ylabel('$\epsilon$')
        subs[0][1].set_ylim(ymin=1e-4)

        subs[1][0].loglog(grid.Rc, model.disc.Stokes()[1])
        subs[1][0].set_xlabel('$R$')
        subs[1][0].set_ylabel('$St$')

        subs[1][1].loglog(grid.Rc, model.disc.grain_size[1])
        subs[1][1].set_xlabel('$R$') 
        subs[1][1].set_ylabel('$a\,[\mathrm{cm}]$')

        l, = subs[0][0].loglog(grid.Rc, model.disc.Sigma_D.sum(0), '--')
        c = l.get_color()
    except AttributeError:
        c = None

    subs[0][0].loglog(grid.Rc, model.disc.Sigma_G, c=c)
    subs[0][0].set_xlabel('$R$')
    subs[0][0].set_ylabel('$\Sigma_\mathrm{G, D}$')
    subs[0][0].set_ylim(ymin=1e-5)

    return [f, subs]

def run(model, io, base_name, verbose=True, n_print=100):

    plot = False
    figs = None
    while not io.finished():
        ti = io.next_event_time()
        while model.t < ti:
            dt = model(ti)

            if verbose and (model.num_steps % n_print) == 0:
                print('Nstep: {}'.format(model.num_steps))
                print('Time: {} yr'.format(model.t / yr))
                print('dt: {} yr'.format(dt / yr))


        if io.check_event(model.t, 'save'):
            if base_name.endswith('.h5'):
                model.dump_hdf5(base_name.format(io.event_number('save')))
            else:
                model.dump_ASCII(base_name.format(io.event_number('save')))

        if io.check_event(model.t, 'plot'):
            plot = True
            err_state = np.seterr(all='warn')

            print('Nstep: {}'.format(model.num_steps))
            print('Time: {} yr'.format(model.t / (2 * np.pi)))
            
            figs = _plot_grid(model, figs)

            np.seterr(**err_state)

        io.pop_events(model.t)

    if plot:
        plt.show()

def main(*args):

    err_state = np.seterr(invalid='raise')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel)

    args = parser.parse_args(*args)

    model = json.load(open(args.model, 'r'))
    
    print("Setting up model")
    time = 0 

    driver = setup_model(model, time)

    output_name, io_control = setup_output(model)

    print("Starting simulation")
    run(driver, io_control, output_name)

    np.seterr(**err_state)

if __name__ == "__main__":
    main() 




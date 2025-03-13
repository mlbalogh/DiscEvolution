# run_model.py
#
# Author: R. Booth
# Date: 4 - Jun - 2018
#
# Run a disc evolution model with transport and absorption / desorption but
# no other chemical reactions. 
#
# Note:
#   The environment variable 
#       "KROME_PATH=/home/rab200/WorkingCopies/krome_ilee/build"
#   should be set.
###############################################################################

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import Msun, AU, yr, Mjup
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar, PhotoStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS
from DiscEvolution.dust import DustGrowthTwoPop, PlanetesimalFormation, FixedSizeDust
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import PlanetDiscDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.internal_photo import EUVDiscAlexander, XrayDiscOwen, XrayDiscPicogna
from DiscEvolution.history import History
import DiscEvolution.photoevaporation as photoevaporation
import DiscEvolution.FRIED.photorate as photorate
from DiscEvolution.planet_formation import *

###############################################################################
# Global Constants
###############################################################################

DefaultModel = "LenzConfig.json"

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

    return disc, grid, eos, gas, Mdot, Mtot

###############################################################################
# Setup Functions
###############################################################################

def setup_disc(model):
    '''Create disc object from initial conditions'''
    grid_model = model['grid']
    disc_model = model['disc']
    disc, grid, eos, gas, Mdot, Mtot = LBP_model(alpha=disc_model['alpha'], rmin=grid_model['R0'], rmax=grid_model['R1'], 
                                                  nr=grid_model['N'], Mdot=grid_model['Mdot'], fixed=False)


def setup_model(model, disc, history, start_time=0, internal_photo_type="Primordial", R_hole=None):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None

    if model['transport']['gas']:
        try:
            gas = ViscousEvolution(boundary=model['grid']['outer_bound'], in_bound=model['grid']['inner_bound'])
        except KeyError:
            print("Default boundaries")
            gas = ViscousEvolution(boundary='Mdot_out')
        
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        dust = SingleFluidDrift(diffuse)
        diffuse = None

    if model['planetesimal']['active']:
        disc._planetesimal = True
        planetesimal_params = model['planetesimal']
        planetesimal = PlanetesimalFormation(disc, d_planetesimal=planetesimal_params['diameter'], St_min=planetesimal_params['St_min'], 
                                             St_max=planetesimal_params['St_max'], pla_eff=planetesimal_params['pla_eff'])

    if model['planets']['active']:
        planet_params = model['planets']
        
        planets = Planets(Nchem = 0)
        
        Rp = planet_params['Rp']
        Mp = planet_params['Mp']

        for i in range(len(Rp)):
            planets.add_planet(0, Rp[i], Mp[i], 0)

        if planet_params['planet_model'] == 'Bitsch2015Model':
            planet_model = Bitsch2015Model(disc, pb_gas_f=0.1)

        planet_model = photorate.PlanetModel(planets['mass'], planets['radius'], planets['a'], planets['ecc'], planets['inc'], planets['f'], planets['t0'])

    return PlanetDiscDriver(disc, gas=gas, dust=dust, planestesimal=planetesimal, diffusion=diffuse, planets=planets, planet_model=planet_model,
                            ext_photoevaporation=photoevap, int_photoevaporation=internal_photo, history=history, t0=start_time)


def setup_output(model):
    
    out = model['output']

    # For explicit control of output times
    if (out['arrange'] == 'explicit'):

        # Setup of the output controller
        output_times = np.arange(out['first'], out['last'], out['interval']) * yr
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)
            output_times = np.insert(output_times,0,0) # Prepend 0

        # Setup of the plot controller
        # Setup of the plot controller
        if out['plot'] and out['plot_times']!=[0]:
            plot = np.array([0]+out["plot_times"]) * yr
        elif out['plot']:
            plot = output_times
        else:
            plot = []

        # Setup of the history controller
        if out['history'] and out['history_times']!=[0]:
            history = np.array([0]+out['history_times']) * yr
        elif out['history']:
            history = output_times
        else:
            history = []

    # For regular, logarithmic output times
    elif (out['arrange'] == 'log'):
        print("Logarithmic spacing of outputs chosen - overrides the anything entered manually for plot/history times.")

        # Setup of the output controller
        if out['interval']<10:
            perdec = 10
        else:
            perdec = out['interval']
        first_log = np.floor( np.log10(out['first']) * perdec ) / perdec
        last_log  = np.floor( np.log10(out['last'])  * perdec ) / perdec
        no_saves = int((last_log-first_log)*perdec+1)
        output_times = np.logspace(first_log,last_log,no_saves,endpoint=True,base=10,dtype=int) * yr
        output_times = np.insert(output_times,0,0) # Prepend 0
        if not np.allclose(out['last'], output_times[-1], 1e-12):
            output_times = np.append(output_times, out['last'] * yr)

        # Setup of the plot controller
        if out['plot']:
            plot = output_times
        else:
            plot = []      

        # Setup of the history controller
        if out['history']:
            history = output_times
        else:
            history = []      

    EC = Event_Controller(save=output_times, plot=plot, history=history)
    
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

    return base_name, EC, output_times / yr


def setup_wrapper(model, restart, output=True):
    # Setup basics
    disc = setup_disc(model)
    if model['disc']['d2g'] > 0:
        dust = True
        d_thresh = model['dust']['radii_thresholds']
    else:
        dust = False
        d_thresh = None
    history = History(dust, d_thresh)

    # Setup model
    if restart:
        disc, history, time, photo_type, R_hole = restart_model(model, disc, history, restart)       
        driver = setup_model(model, disc, history, time, internal_photo_type=photo_type, R_hole=R_hole)
    else:
        driver = setup_model(model, disc, history)

    # Setup outputs
    if output:
        output_name, io_control, output_times = setup_output(model)
        plot_name = model['output']['plot_name']
    else:
        output_name, io_control, plot_name = None, None, None

    # Truncate disc at base of wind
    if driver.photoevaporation_external and not restart:
        if (isinstance(driver.photoevaporation_external,photoevaporation.FRIEDExternalEvaporationMS)):
            driver.photoevaporation_external.optically_thin_weighting(disc)
            optically_thin = (disc.R > driver.photoevaporation_external._Rot)
        else:
            initial_trunk = photoevaporation.FRIEDExternalEvaporationMS(disc)
            initial_trunk.optically_thin_weighting(disc)
            optically_thin = (disc.R > initial_trunk._Rot)

        disc._Sigma[optically_thin] = 0

        """Lines to truncate with no mass loss if required for direct comparison"""
    """else:
        photoevap = photoevaporation.FRIEDExternalEvaporationMS(disc)
        optically_thin = (disc.R > disc.Rot(photoevap))"""
    
    Dt_nv = np.zeros_like(disc.R)
    if driver.photoevaporation_external:
        # Perform estimate of evolution for non-viscous case
        (_, _, M_cum, Dt_nv) = driver.photoevaporation_external.get_timescale(disc)

    return disc, driver, output_name, io_control, plot_name, Dt_nv


def restart_model(model, disc, history, snap_number):
    # Resetup model
    out = model['output']
    reader = DiscReader(out['directory'], out['base'], out['format'])

    snap = reader[snap_number]

    disc.Sigma[:] = snap.Sigma
    try:
        disc.dust_frac[:] = snap.dust_frac
        disc.grain_size[:] = snap.grain_size
    except:
        pass

    time = snap.time * yr       # Convert real time (years) to code time

    disc.update(0)

    # Revise and write history
    infile = model['output']['directory']+"/"+"discproperties.dat"
    history.restart(infile, snap_number)

    # Find current location of hole, if appropriate
    try:
        R_hole = history._Rh[-1]
        if np.isnan(R_hole):
            R_hole = None
        else:
            print("Hole is at: {} AU".format(R_hole))
    except:
        R_hole = None

    return disc, history, time, snap.photo_type, R_hole     # Return disc objects, history, time (code units), input data and internal photoevaporation type

###############################################################################
# Saving - now moved to the history module
###############################################################################

###############################################################################
# Run
###############################################################################    

def run(model, io, base_name, all_in, restart, verbose=True, n_print=1000, end_low=False):
    external_mass_loss_mode = all_in['fuv']['photoevaporation']

    save_no = 0
    end = False     # Flag to set in order to end computation
    first = True    # Avoid duplicating output during hole clearing
    hole_open = 0   # Flag to set to snapshot hole opening
    hole_save = 0   # Flag to set to snapshot hole opening
    if all_in['transport']['radial drift']:
        hole_snap_no = 1e5
    else:
        hole_snap_no = 1e4
    hole_switch = False

    if restart:
        # Skip evolution already completed
        while not io.finished():
            ti = io.next_event_time()
            
            if ti > model.t:
                break
            else:
                io.pop_events(model.t)

    while not io.finished():
        ti = io.next_event_time()
        while (model.t < ti and end==False):
            """
            External photoevaporation - if present, model terminates when all cells at (or below) the base rate as unphysical (and prevents errors).
            Internal photoevaporation - if present, model terminates once the disc is empty.
            Accretion - optionally, the model terminates once unobservably low accretion rates (10^-11 solar mass/year)
            """

            # External photoevaporation -  Read mass loss rates
            if model.photoevaporation_external:
                not_empty = (model.disc.Sigma_G > 0)
                Mdot_evap = model.photoevaporation_external.mass_loss_rate(model.disc,not_empty)
                # Stopping condition
                if (np.amax(Mdot_evap)<=1e-10):
                    print ("Photoevaporation rates below FRIED floor... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                elif external_mass_loss_mode == 'Constant' and model.photoevaporation_external._empty:
                    print ("Photoevaporation has cleared entire disc... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True                

            # Internal photoevaporation
            if model._internal_photo:
                # Stopping condition
                if model.photoevaporation_internal._empty:
                    print ("No valid Hole radius as disc is depleted... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                # Check if need to reset the hole or if have switched to direct field
                elif model.photoevaporation_internal._Thin and not hole_switch:
                    hole_open = np.inf
                    hole_switch = True
                elif model.photoevaporation_internal._reset:
                    hole_open = 0
                    model.photoevaporation_internal._reset = False
                    model.history.clear_hole()
                # If the hole has opened, count steps and determine whether to do extra snapshot
                if model.photoevaporation_internal._Hole:
                    hole_open += 1
                    if (hole_open % hole_snap_no) == 1 and not first:
                        ti = model.t
                        break

            # Viscous evolution - Calculate accretion rate
            if model.gas and end_low:
                M_visc_out = 2*np.pi * model.disc.grid.Rc[0] * model.disc.Sigma[0] * model._gas.viscous_velocity(model.disc)[0] * (AU**2)
                Mdot_acc = -M_visc_out*(yr/Msun)
                # Stopping condition
                if (Mdot_acc<1e-11):
                    print ("Accretion rates below observable limit... terminating calculation at ~ {:.0f} yr".format(model.t/yr))
                    end = True
                    
            if end:
                ### Stop model ###
                last_save=0
                last_plot=0
                last_history=0
                # If there are save times left
                if np.size(io.event_times('save'))>0:
                    last_save = io.event_times('save')[-1]
                # If there are plot times left 
                if np.size(io.event_times('plot'))>0:
                    last_plot = io.event_times('plot')[-1]
                # If there are history times left 
                if np.size(io.event_times('history'))>0:
                    last_history = io.event_times('history')[-1]
                # Remove all events up to the end
                last_t = max(last_save,last_plot,last_history)
                io.pop_events(last_t)

            else:
                ### Evolve model and return timestep ###
                dt = model(ti)
                first = False

            ### Printing
            if verbose and (model.num_steps % n_print) == 0:
                print('Nstep: {}'.format(model.num_steps))
                print('Time: {} yr'.format(model.t / yr))
                print('dt: {} yr'.format(dt / yr))
                if model.photoevaporation_internal and model.photoevaporation_internal._Hole:
                    print("Column density to hole is N = {} g cm^-2".format(model._internal_photo._N_hole))
                    print("Empty cells: {}".format(np.sum(model.disc.Sigma_G<=0)))
                
        grid = model.disc.grid
        
        ### Saving
        if io.check_event(model.t, 'save') or end or (hole_open % hole_snap_no)==1:
            # Print message to record this
            if (hole_open % hole_snap_no)==1:
                print ("Taking extra snapshot of properties while hole is clearing")
                hole_save+=1
            elif end:
                print ("Taking snapshot of final disc state")
            else:
                print ("Making save at {} yr".format(model.t/yr))
            if base_name.endswith('.h5'):
                    model.dump_hdf5( base_name.format(save_no))
            else:
                    model.dump_ASCII(base_name.format(save_no))
            save_no+=1
        if io.check_event(model.t, 'history') or end or (hole_open % hole_snap_no)==1:
            # Measure disc properties and record
            model.history(model)
            # Save state
            model.history.save(model,all_in['output']['directory'])

        io.pop_events(model.t)


def main():
    # Retrieve model from inputs
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=DefaultModel, help='specify the model input json file')
    parser.add_argument("--restart", "-r", type=int, default=0, help='specify a save number from which to restart')
    parser.add_argument("--end", "-e", action="store_true", help='include in order to stop when below observable accretion rates')
    args = parser.parse_args()
    model = json.load(open(args.model, 'r'))
    
    # Do all setup
    disc, driver, output_name, io_control, plot_name, Dt_nv = setup_wrapper(model, args.restart)

    # Run model
    run(driver, io_control, output_name, model, args.restart, end_low=args.end)
        
    # Save disc properties
    outputdata = driver.history.save(driver,model['output']['directory'])

if __name__ == "__main__":
    main()
    

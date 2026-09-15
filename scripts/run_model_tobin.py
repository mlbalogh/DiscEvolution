## import dependencies
import os
import json
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler

## Add the path to the DiscEvolution directory
sys.path.append('/Users/ben/Downloads/Planet Formation/Code/DiscEvolution Core Accretion')

## import DiscEvolution modules
from DiscEvolution.constants import *
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import *
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution, HybridWindModel, TaboneSolution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import *
from DiscEvolution.dust import PlanetesimalFormation
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.chemistry import *

start_time = time.time()
plt.rcParams.update({'font.size': 16})

def run_model(config):
    """
    Run the disk evolution model and plot the results.
    
    Parameters:
        config (dict): Configuration dictionary containing all parameters.
    """

    ## Extract parameters from config
    grid_params = config['grid']
    sim_params = config['simulation']
    star_params = config['star']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planets']
    chemistry_params = config["chemistry"]
    planetesimal_params = config['planetesimal']
    wind_params = config["winds"]

    ## -----------
    ## Set up disc
    ## -----------

    ## Create grid
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing = grid_params['spacing'])

    ## Create star
    star = SimpleStar(M = star_params["M"], R = star_params["R"], T_eff = star_params['T_eff'])

    ## Create time array
    if sim_params['t_interval'] == "power":
        ## Determine the number of points needed
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            times = np.logspace(0, np.log10(sim_params['t_final']), num = num_points) * 2 * np.pi

        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            times = np.logspace(np.log10(sim_params['t_initial']), np.log10(sim_params['t_final']), num = num_points) * 2 * np.pi

    elif type(sim_params['t_interval']) == list:
        times = np.array(sim_params['t_interval']) * 2 * np.pi * 1e6
    
    else:
        times = np.arange(sim_params['t_initial'], sim_params['t_final'], sim_params['t_interval']) * 2 * np.pi

    ## Define opacity class used, if not Tazzari, defaults to Zhu in IrradiatedEOS
    if eos_params["opacity"] == "Tazzari":
        kappa = Tazzari2016()

    elif eos_params["opacity"] == "Zhu2012":
        kappa = None

    else:
        kappa = None
    
    if grid_params['type'] == 'Booth-alpha':
        ## For fixed Rd, Mdot and Mdisk, solve for alpha
    
        ## Extract params
        Mdot = disc_params['Mdot']
        Mdisk = disc_params['M']
        alpha = disc_params['alpha']
        Rd = disc_params['Rd']
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma = 1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd / R) * np.exp(-R / Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2) / Msun)
            return Sigma
    
        ## Define an initial guess for the Sigma profile
        Sigma = Sigma_profile(R, Rd, Mdisk)
    
        ## Define a gas class, to be used later
        gas_temp = ViscousEvolutionFV()

        ## Iterate to get alpha
        for j in range(100):
            ## Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t = alpha)

            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)

            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t = alpha, kappa = kappa, Tmax = eos_params["Tmax"])
        
            ## Update EOS with current sigma profile
            eos.set_grid(grid)
            eos.update(0, Sigma)

            ## Define a disc given current EOS and Sigma
            disc = AccretionDisc(grid, star, eos, Sigma)

            ## Find the current Mdot in the disc
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, Sigma))

            ## Scale Sigma by Mdot to get desired Mdot
            Sigma_new = Sigma * Mdot / Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            ## Define new disc given new Sigma profile
            disc = AccretionDisc(grid, star, eos, Sigma)

            ## Scale alpha by Mdisk so that desired disk mass is achieved
            alpha= alpha*(disc.Mtot() / Msun) / Mdisk

            if grid_params["smart_bining"]:
                ## If using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing = grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

    elif grid_params['type'] == 'Booth-Rd':
        ## For fixed alpha, Mdot and Mdisk, solve for Rd
    
        ## Extract params
        Mdot = disc_params['Mdot']
        Mdisk = disc_params['M']
        alpha = disc_params['alpha']
        Rd = disc_params['Rd'] # initial guess
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma = 1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd / R) * np.exp(-R / Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2) / Msun)
            return Sigma
    
        ## Create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        ## Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t = alpha)

        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)

        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t = alpha, kappa = kappa, Tmax = eos_params["Tmax"])
        
        ## Update EOS with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        ## Define gas class to be used in first iteration
        gas_temp = ViscousEvolutionFV()

        ## iterate to get Rd
        for j in range(100):
            ## Initialize a disc with current Sigma and EOS
            disc = AccretionDisc(grid, star, eos, Sigma)

            ## Find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S = Sigma))

            ## Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma * Mdot / Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new= Rd*np.sqrt(Mdisk/(disc.Mtot() / Msun))
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            ## Define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            ## Update EOS with new Sigma to have correct temperature profile
            eos.update(0, Sigma)
    
    elif grid_params['type'] == "LBP":
        ## Define viscous evolution to calculate drift velocity later
        gas = ViscousEvolutionFV()

        ## Extract parameters
        gamma = disc_params['gamma']
        R = grid.Rc
        Rd = disc_params['Rd']
        Mdot = disc_params['Mdot']* Msun / yr 
        Mdisk = disc_params['M']* Msun
        alpha = disc_params['alpha']
        mu = chemistry_params['mu']
        rin = R[0]
        xin = R[0] / Rd

        ## Calculate the keplerian velocity
        fin = np.exp(-xin ** (2. - gamma)) * (1. - 2. * (2. - gamma) * xin ** (2. - gamma))
        nud_goal = (Mdot / Mdisk) * (2. * Rd * Rd) / (3. * (2. - gamma)) / fin * AU * AU # cm^2
        nud_cgs = nud_goal * yr / 3.15e7
        Om_invsecond = star.Omega_k(Rd) * yr / 3.15e7

        ## Calculate initial sound speed and temperature profile
        cs0 = np.sqrt(Om_invsecond * nud_cgs / alpha) # cm/s
        Td = cs0 * cs0 * mu * m_p / k_B # KT = Td * (R / Rd) ** (gamma - 1.5)
        T = Td * (R / Rd) ** (gamma - 1.5)

        ## Calculate the actual sound speed and surface density profile
        cs = np.sqrt(GasConst * T / mu) # cgs
        cs0 = np.sqrt(GasConst * Td / mu) # cgs
        nu = alpha * cs * cs / (star.Omega_k(R) * yr / 3.15e7) # cm2/s
        nud = np.interp(Rd, grid.Rc, nu) * 3.15e7 / yr # cm^2 
        Sigma = LBP_Solution(Mdisk, Rd * AU, nud, gamma = gamma)
        Sigma0 = Sigma(R * AU, 0) 

        ## Adjust alpha so initial Mdot is correct
        for i in range(10):
            ## Define an EOS
            eos = IrradiatedEOS(star, alpha_t = disc_params['alpha'], kappa = kappa, Tmax = eos_params["Tmax"])
            eos.set_grid(grid)
            eos.update(0, Sigma0)

            ## Define a temporary disc to compute Mdot
            disc = AccretionDisc(grid, star, eos, Sigma0)

            ## Ddjust alpha depending on current Mdot and wanted Mdot
            vr = gas.viscous_velocity(disc, Sigma0)
            Mdot_actual = disc.Mdot(vr[0]) # * (Msun / yr)
            alpha = alpha * (Mdot / Msun * yr) / Mdot_actual
        Sigma = Sigma0

    elif grid_params['type'] == 'Booth-Mdot':
        ## For fixed alpha, Rd, and Mdisk, solve for Mdot
    
        ## Extract parameters
        R = grid.Rc
        Rd = disc_params['Rd']
        Mdot = disc_params['Mdot'] * Msun / yr # initial guess
        Mdisk = disc_params['M']
        alpha = disc_params['alpha']

        ## Define Sigma profile, scale by Mdisk to get correct disk mass
        Sigma = (Rd / R) * np.exp(-R / Rd)
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2) / Msun)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t = alpha)

        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)

        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t = alpha, kappa = kappa, Tmax = eos_params["Tmax"])
        
        ## Update the EOS with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)

    elif grid_params['type'] == 'winds-alpha':
        ## For fixed Rd, Mdot and Mdisk, solve for alpha with disk winds
        ## Assumes gamma = 1

        ## extract params
        Mdot = disc_params['Mdot'] # solar masses per year
        Mdisk = disc_params['M']* Msun
        psi = wind_params['psi_DW']
        # lambda_DW = wind_params['lambda_DW']
        Rd = disc_params['Rd']
        alpha = disc_params['alpha']
        e_rad = wind_params["e_rad"]
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha / (1 + psi)

        ## Initial guess for Sigma
        Sigma_d = Mdisk / (2 * np.pi * (Rd * AU) ** 2)
        # xi = 0.25 * (1 + psi) * (np.sqrt(1 + 4 * psi / ((lambda_DW - 1) * (psi + 1)**2)) - 1)
        xi = 0
        Sigma = Sigma_d * (R / Rd) ** (xi - gamma) * np.exp(-(R / Rd) ** (2 - gamma))

        ## Define an initial disc and gas class to be used later
        disc = AccretionDisc(grid, star, eos = None, Sigma = Sigma)
        gas_temp = HybridWindModel(psi, lambda_DW)

        ## Scale Sigma by current Mtot just in case Sigma is not quite at the correct value to have the desired Mdisk (which often happens)
        Mtot = disc.Mtot()
        Sigma[:] *= Mdisk / Mtot

        for i in range(100):
            ## Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t = alpha_SS)

            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)

            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t = alpha_SS, kappa = kappa, psi = psi, e_rad = e_rad, Tmax = eos_params["Tmax"])
            
            ## Update EOS with grid and Sigma
            eos.set_grid(grid)
            eos.update(0, Sigma)

            ## Define new disc 
            disc = AccretionDisc(grid, star, eos, Sigma)

            ## Find current Mdot in the disc given Sigma and current EOS
            vr = gas_temp.viscous_velocity(disc, Sigma)
            Mdot_actual = disc.Mdot(vr)[0] # solar masses per year

            # Scale alpha by Mdot
            alpha_new = alpha * Mdot / Mdot_actual
            alpha = 0.5 * (alpha + alpha_new) # average done to damp oscillations in numerical solution

            ## Find a new alpha_SS given new alpha.
            alpha_SS = alpha / (1 + psi)

            if grid_params["smart_bining"]:
                ## If using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing = grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

    elif grid_params['type'] == 'winds-Rd':
        ## For fixed alpha, Mdot and Mdisk, solve for Rd with disk winds
    
        ## Extract params
        Mdot = disc_params['Mdot'] # solar masses per year
        Mdisk = disc_params['M'] * Msun
        psi = wind_params['psi_DW']
        # lambda_DW = wind_params['lambda_DW']
        Rd = disc_params['Rd']
        alpha = disc_params['alpha']
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha / (1 + psi)

        def Sigma_profile(R, Rd, Mdisk):
            """Creates a non-steady state Sigma profile for gamma = 1, scaled such that the disk mass equals Mdisk"""
            chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4 * psi / ((lambda_DW - 1) * (psi + 1) ** 2)) - 1)
            Sigma = (R / Rd) ** (chi - gamma) * np.exp(-(R / Rd) ** (2 - gamma))
            Sigma *= Mdisk / np.trapezoid(Sigma, np.pi * (R * AU) ** 2)
            return Sigma
    
        ## Create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        ## Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t = alpha_SS)

        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)

        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t = alpha_SS, kappa = kappa, Tmax = eos_params["Tmax"])
        
        ## Update EOS with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        ## Define gas class to be used in first iteration
        gas_temp = HybridWindModel(psi, lambda_DW)

        ## Iterate to get Rd
        for j in range(100):
            ## Initialize a disc with current Sigma and EOS
            disc = AccretionDisc(grid, star, eos, Sigma)

            ## Find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S = Sigma))

            ## Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            ## Define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new = Rd * np.sqrt(Mdisk / disc.Mtot())
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            ## Define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            ## Update EOS with new Sigma to have correct temperature profile
            eos.update(0, Sigma)

    elif grid_params['type'] == 'winds-Mdot':
        ## For fixed alpha, Rd, and Mdisk, solve for Mdot with disk winds included
    
        ## Extract parameters
        R = grid.Rc
        Rd = disc_params['Rd']
        Mdot = disc_params['Mdot'] # initial guess
        Mdisk = disc_params['M']
        alpha = disc_params['alpha']
        psi = wind_params['psi_DW']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1 
        gamma = disc_params['gamma']
        alpha_SS = alpha / (1 + psi)

        ## Define Sigma profile, scale by Mdisk to get correct disk mass
        chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4 * psi / ((lambda_DW - 1) * (psi + 1) ** 2)) - 1)
        Sigma = (R / Rd) ** (chi - gamma) * np.exp(-(R / Rd) ** (2 - gamma))
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU) ** 2) / Msun)

        ## Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t = alpha_SS)

        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)

        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t = alpha_SS, kappa = kappa, psi = psi, e_rad = e_rad, Tmax = eos_params["Tmax"])
        
        ## Update the EOS with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)

    ## ---------------
    ## Set up dynamics
    ## ---------------

    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW)

        else:
            gas = ViscousEvolutionFV()
    
    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc = disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        dust = SingleFluidDrift(diffusion = diffuse, settling = dust_growth_params['settling'], van_leer = transport_params['van_leer'])
        diffuse = None

    ## --------------
    ## Set disc model
    ## --------------

    try:
        disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], disc_params['d2g_SI'],
            Sigma = Sigma, feedback = dust_growth_params["feedback"], Sc = disc_params["Sc"],
            f_ice = dust_growth_params['f_ice'], thresh = dust_growth_params['thresh'],
            uf_0 = dust_growth_params["uf_0"], uf_ice = dust_growth_params["uf_ice"], gas = gas,
            rho_s = dust_growth_params['rho_s'])
        
    except Exception as e:
        raise e
    
    ## ----------------
    ## Set up chemistry
    ## ----------------

    disc.chem = None
    if chemistry_params["on"]:

        ## Extract params
        N_cell = grid_params["nr"]

        ## Choose chemical model
        if chemistry_params["chem_model"] == "Simple":
            chemistry = SimpleCOChemOberg()

        elif chemistry_params["chem_model"] == "Equilibrium":
            chemistry = EquilibriumCOChemOberg(a = 1e-5, fix_ratios = True)

        elif chemistry_params["chem_model"] == "TimeDep":
            chemistry = TimeDepCOChemOberg(a = 1e-5)

        else:
            raise Exception("Valid chemistry model not selected. Choose Simple, Equilibrium, or TimeDep.")
        
        ## Setup the dust-to-gas ratio from the chemistry
        X_solar = SimpleCOAtomAbund(N_cell) # data array containing abundances of atoms
        X_solar.set_solar_abundances() # redefines abundances by multiplying with specific constants

        ## Iterate ice fractions to get the dust-to-gas ratio
        for i in range(100):
            if chemistry_params["assert_d2g"]:

                ## Find the total gas and dust mass
                M_dust = np.trapezoid(disc.Sigma_D.sum(0), np.pi * grid.Rc ** 2)
                M_gas = np.trapezoid(disc.Sigma_G, np.pi * grid.Rc ** 2)

                ## Calculate a modification fraction by dividing the wanted dust fraction by the current dust fraction
                mod_frac = disc_params["d2g"] / (M_dust / M_gas)

                ## Multiply modification fraction into dust fraction to assert wanted dust fraction
                disc.dust_frac[:] = disc.dust_frac * mod_frac

            dust_frac = disc.dust_frac.sum(0)
            
            ## Returns MolecularIceAbund class containing SimpleCOMolAbund for gas and ice
            chem = chemistry.equilibrium_chem(disc.T, disc.midplane_gas_density, dust_frac, X_solar)
            disc.initialize_dust_density(chem.ice.total_abund)
        disc.chem = chem

        disc.update_ices(disc.chem.ice)
        
    ## --------------
    ## Set up planets
    ## --------------

    time_keeper = []

    if planet_params['include_planets']:
        if chemistry_params["on"]:
            Nchem = disc.chem.ice.data.shape[0]
            planets = Planets(Nchem = Nchem)

        else:
            planets = Planets(Nchem = 0)

        planet_model = Bitsch2015Model(
            disc,
            pb_gas_f = planet_params["pb_gas_f"],
            f_plt = planet_params["f_plt"],
            migrate = planet_params["migrate"],
            pebble_acc = planet_params["pebble_accretion"],
            gas_acc = planet_params["gas_accretion"],
            planetesimal_acc_migrate = planet_params["planetesimal_accretion_migrate"],
            planetesimal_acc_insitu = planet_params["planetesimal_accretion_insitu"],
            winds = wind_params["on"],
            rho_core = planet_params["rho_core"],)
        
        planet_model.set_disc(disc)

        Mp = planet_params['Mp']
        Rp = planet_params['Rp']

        Rs, Mcs, Mes, X_cores, X_envs, disk_Mdot_p = [], [], [], [], [], []
        Mdot_planetesimal, Mdot_pebble_core, Mdot_pebble_env, Mdot_migration, Mdot_gas = [], [], [], [], []
        M_iso_planetesimal, M_iso_pebble = [], []

        ## Keeps the per-planet tracking lists index-aligned with 'planets'
        ## Planets insrted after t = 0 are backfilled with zeros to keep the lists aligned
        def add_planet_tracking_slot():
            n_backfill = len(time_keeper)

            Rs.append([np.nan] * n_backfill)
            Mcs.append([np.nan] * n_backfill)
            Mes.append([np.nan] * n_backfill)
            disk_Mdot_p.append([np.nan] * n_backfill)

            Mdot_planetesimal.append([np.nan] * n_backfill)
            Mdot_pebble_core.append([np.nan] * n_backfill)
            Mdot_pebble_env.append([np.nan] * n_backfill)
            Mdot_migration.append([np.nan] * n_backfill)
            Mdot_gas.append([np.nan] * n_backfill)

            M_iso_planetesimal.append([np.nan] * n_backfill)
            M_iso_pebble.append([np.nan] * n_backfill)

            if chemistry_params["on"]:
                X_cores.append([[np.nan] * n_backfill for num in range(0, Nchem, 1)])
                X_envs.append([[np.nan] * n_backfill for num in range(0, Nchem, 1)])

        ## Planets with Mp = "SI" are are only inserted once the planetesimal surface density becomes nonzero
        pending_SI_planets = []

        for i in range(len(Rp)):
            t_impl = planet_params["implant_time"][i]
            R_impl = Rp[i]
            M_impl = Mp[i]

            if str(M_impl).upper() == "SI":
                pending_SI_planets.append((R_impl, M_impl))

            else:
                planet_model.insert_new_planet(t_impl, R_impl, M_impl, planets)
                add_planet_tracking_slot()

    else:
        planets = None

    disk_Mdot_star, disk_Mass, Tc, Sigc = [], [], [], []

    ## --------------------
    ## Set up planetesimals
    ## --------------------

    disc._planetesimal = None
    if planetesimal_params['active']:
        disc._planetesimal = PlanetesimalFormation(
            disc, planets,
            d_planetesimal = planetesimal_params['diameter'],
            rho_pltsml = planetesimal_params['rho_pltsml'],
            St_min = planetesimal_params['St_min'],
            St_max = planetesimal_params['St_max'],
            pla_eff = planetesimal_params['pla_eff'],
            drag = planetesimal_params['drag'],
            VS_embryo = planetesimal_params['VS_embryo'],
            VS_pltsml = planetesimal_params['VS_pltsml'],
            DF = planetesimal_params['DF'],
            e_init = planetesimal_params['e_init'],
            i_init = planetesimal_params['i_init'])
        
    ## ----------------
    ## First data point
    ## ----------------

    disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
    disk_Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU * AU) * (yr / Msun)
    disk_Mdot_star.append(disk_Mdot[0])
    disk_Mass.append(disc.Mtot())
    Tc.append(disc.T[0])
    Sigc.append(disc.Sigma[0])

    if planet_params['include_planets']:
        ## Exact rates at t = 0, before any integrate() call has populated planet_model.rates

        if chemistry_params["on"]:
            M_Z_0   = (planets.X_core * planets.M_core).sum(0) + (planets.X_env * planets.M_env).sum(0)
            M_HHe_0 = planets.M_core + planets.M_env - M_Z_0

        else:
            M_Z_0, M_HHe_0 = planets.M_core, planets.M_env

        initial_rates = planet_model._growth_rates(planets.R, planets.M_core, planets.M_env, M_Z_0, M_HHe_0)

        for count, planet in enumerate(planets):
            Rs[count].append(planet.R.copy())
            Mcs[count].append(planet.M_core.copy())
            Mes[count].append(planet.M_env.copy())
            disk_Mdot_p[count].append(np.interp(planet.R,grid.Rc[0:-1], disk_Mdot))

            if planet_params["planetesimal_accretion_insitu"]:
                Mdot_planetesimal[count].append(initial_rates["Mdot_planetesimal_insitu"][count] * yr)
                M_iso_planetesimal[count].append(planet_model._pla_acc.M_iso_pltsml(planet.R))

            if planet_params["pebble_accretion"]:
                Mdot_pebble_core[count].append(initial_rates["Mdot_pebble_core"][count] * yr)
                Mdot_pebble_env[count].append(initial_rates["Mdot_pebble_env"][count] * yr)
                M_iso_pebble[count].append(planet_model._peb_acc.M_iso(planet.R))

            if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                Mdot_migration[count].append(initial_rates["Mdot_planetesimal_migration"][count] * yr)

            if planet_params["gas_accretion"]:
                Mdot_gas[count].append(initial_rates["Mdot_gas"][count] * yr)

            if chemistry_params["on"]:
                for count2, chem in enumerate(planet.X_core):
                    X_cores[count][count2].append(chem)
                    X_envs[count][count2].append(planet.X_env[count2])

    time_keeper.append(0)

    ## -------------
    ## Prepare plots
    ## -------------

    fig, axes = plt.subplots(2, 2, figsize = (16, 12))
    axes = axes.flatten()
    
    ## Find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(vr[0])
        
    ## Display disk characteristics
    plt.figtext(0.5, 0, f"Mdot = {Mdot:.3e}, alpha = {disc._eos._alpha_t:.3e}, Mtot = {disc.Mtot() / Msun:.3e}, Rd = {disc.RC():.3e}", ha = "center")

    ## This is to synchronize colors
    d = 0 
    # colors = ["black", "red", "green", "blue", "cyan"]
    nplanets = len(config["planets"]["Mp"])
    colors = [cm.viridis(i / nplanets) for i in range(nplanets)]

    cm2 = plt.get_cmap("viridis")

    ## Gradient colors also present to give options
    n_times = len(times)
    color1 = iter(plt.cm.Blues(np.linspace(0.4, 1, n_times)[::-1]))
    color2 = iter(plt.cm.Greys(np.linspace(0.4, 1, n_times)[::-1]))
    color3 = iter(plt.cm.Greens(np.linspace(0.4, 1, n_times)[::-1]))
    color4 = iter(plt.cm.Reds(np.linspace(0.4, 1, n_times)[::-1]))

    ## ---------
    ## Run model
    ## ---------

    t = 0
    n = 0
    data = {}
    data["R"] = grid.Rc.copy().tolist()
    data["time_snap"] = []
    data["Sigma_G"] = []
    data["Sigma_dust"] = []
    data["Sigma_pebbles"] = []
    data["Vdrift_grains"] = []
    data["Vdrift_pebbles"] = []
    data["St_grains"] = []
    data["St_pebbles"] = []
    data["T"] = []

    if planetesimal_params['active']:
        data["Sigma_planetesimals"] = []
        data["St_planetesimals"] = []
        data["e_planetesimals"] = []
        data["i_planetesimals"] = []

        if planetesimal_params['drag'] or planetesimal_params['VS_embryo'] or planetesimal_params['VS_pltsml'] or planetesimal_params['DF']:
            data["de2_dt"] = []
            data["di2_dt"] = []

        if planetesimal_params['drag']:
            data["de2_dt_drag"] = []
            data["di2_dt_drag"] = []

        if planetesimal_params['VS_embryo']:
            data["de2_dt_VS_embryo"] = []
            data["di2_dt_VS_embryo"] = []

        if planetesimal_params['VS_pltsml']:
            data["de2_dt_VS_pltsml"] = []
            data["di2_dt_VS_pltsml"] = []

        if planetesimal_params['DF']:
            data["de2_dt_DF"] = []
            data["di2_dt_DF"] = []

    if alpha_SS > 5e-3:
        print ("Not Running model - alpha too high. Alpha, Rd, Mdisk = ", eos.alpha, Rd, disc.Mtot() / Msun)

    else:   
        print ("Running model. Alpha, Rd, Mdisk = ", eos.alpha, Rd, disc.Mtot() / Msun)
        for ti in times:
            while t < ti:
                ## Find timestep given gas and dust maximum timesteps
                dt = ti - t
                if transport_params['gas_transport']:
                    dt = min(dt, disc._gas.max_timestep(disc))

                if transport_params['radial_drift']:
                    dt = min(dt, dust.max_timestep(disc))
                
                ## Extract updated dust frac to update gas
                dust_frac = None

                try:
                    dust_frac = disc.dust_frac

                except AttributeError:
                    pass

                ## Extract gas tracers
                gas_chem, ice_chem = None, None

                try:
                    gas_chem = disc.chem.gas.data
                    ice_chem = disc.chem.ice.data

                except AttributeError:
                    pass

                ## Do gas evolution
                if transport_params['gas_transport']:
                    ## To preserve planetesimal surface density so that it doesn't move with a change in Sigma as a whole, we do the following
                    if disc._planetesimal:
                        disc._gas(dt, disc, [dust_frac[:-1], gas_chem, ice_chem])

                    else:
                        disc._gas(dt, disc, [dust_frac, gas_chem, ice_chem])

                ## Update planetesimals
                if disc._planetesimal:
                    disc._planetesimal.update(dt, disc, dust)

                ## Insert any pending "SI" planets once the planetesimal surface density is nonzero
                if planet_params['include_planets'] and pending_SI_planets and disc._planetesimal:
                    still_pending = []

                    for R_impl, M_impl in pending_SI_planets:
                        if disc.interp(R_impl, disc.Sigma_D[2]) > 0:
                            planet_model.insert_new_planet(t, R_impl, M_impl, planets)
                            add_planet_tracking_slot()

                        else:
                            still_pending.append((R_impl, M_impl))

                    pending_SI_planets = still_pending

                ## Do dust evolution
                if transport_params['radial_drift']:
                    dust(dt, disc, gas_tracers = gas_chem, dust_tracers = ice_chem)

                if diffuse is not None:
                    if gas_chem is not None:
                        gas_chem[:] += dt * diffuse(disc, gas_chem)

                    if ice_chem is not None:
                        ice_chem[:] += dt * diffuse(disc, ice_chem) # may use planetesimals to move, double check

                    if dust_frac is not None:
                        if disc._planetesimal:
                            ## Excluding planetesimals (assume they don't move)
                            dust_frac[:2] += dt * diffuse(disc, dust_frac[:2]) 

                        else: 
                            dust_frac[:] += dt * diffuse(disc, dust_frac[:])

                ## Pin the values to >= 0 and <=1
                disc.Sigma[:] = np.maximum(disc.Sigma, 0)     
                disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
                disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)

                if chemistry_params["on"]:
                    disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                    disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

                if chemistry_params["on"]:
                    ## Exclude planetesimals from chemistry (assume they don't chemically interact with the disc)
                    if disc._planetesimal:
                        chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac[:-1].sum(0), disc.chem)

                    else:
                        chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac.sum(0), disc.chem)
                    
                    disc.update_ices(disc.chem.ice)

                if planet_params['include_planets']:
                    ## Update the planet masses and radii
                    planet_model.integrate(dt, planets) 

                ## Update disc
                disc.update(dt)
                
                ## Increase time and go forward a step
                t += dt
                n += 1

                if (n % 1000) == 0:
                    print('\rNstep: {}'.format(n), flush = "True")
                    # print('\rTime: {} Myr'.format(t / (1.e6* 2 * np.pi)), end = "", flush = "True")
                    print('\rTime: {} Myr'.format(t / (1.e6 * 2 * np.pi)), flush = "True")
                    print('\rdt: {} yr'.format(dt / (2 * np.pi)), flush = "True")
                
                if n % 5 == 0:
                    time_keeper.append(t / (2 * np.pi))

                    disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
                    disk_Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v* (AU * AU) * (yr / Msun)
                    disk_Mdot_star.append(disk_Mdot[0])
                    disk_Mass.append(disc.Mtot())
                    Tc.append(disc.T[0])
                    Sigc.append(disc.Sigma[0])

                if planet_params['include_planets']:
                    ## Collect data for planet growth track and chemistry
                    if n % 5 == 0:
                        for count, planet in enumerate(planets):
                            Rs[count].append(planet.R.copy())
                            Mcs[count].append(planet.M_core.copy())
                            Mes[count].append(planet.M_env.copy())
                            disk_Mdot_p[count].append(np.interp(planet.R, grid.Rc[0:-1], disk_Mdot))

                            ## planet_model.rates holds the exact rates that drove the most recent integrate() call
                            if planet_params["planetesimal_accretion_insitu"]:
                                Mdot_planetesimal[count].append(planet_model.rates["Mdot_planetesimal_insitu"][count] * yr)
                                M_iso_planetesimal[count].append(planet_model._pla_acc.M_iso_pltsml(planet.R))

                            if planet_params["pebble_accretion"]:
                                Mdot_pebble_core[count].append(planet_model.rates["Mdot_pebble_core"][count] * yr)
                                Mdot_pebble_env[count].append(planet_model.rates["Mdot_pebble_env"][count] * yr)
                                M_iso_pebble[count].append(planet_model._peb_acc.M_iso(planet.R))

                            if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                                Mdot_migration[count].append(planet_model.rates["Mdot_planetesimal_migration"][count] * yr)

                            if planet_params["gas_accretion"]:
                                Mdot_gas[count].append(planet_model.rates["Mdot_gas"][count] * yr)

                            if chemistry_params["on"]:
                                for count2, chem in enumerate(planet.X_core):
                                    X_cores[count][count2].append(chem)
                                    X_envs[count][count2].append(planet.X_env[count2])

            data["time_snap"].append(t / (2 * np.pi * 1e6)) # Myr
            data["Sigma_G"].append(disc.Sigma_G.copy().tolist())
            data["Sigma_dust"].append(disc.Sigma_D[0].copy().tolist())
            data["Sigma_pebbles"].append(disc.Sigma_D[1].copy().tolist())

            v_drift = disc.v_drift.copy()
            stokes = disc.Stokes().copy()

            data["Vdrift_grains"].append(v_drift[0].tolist())
            data["Vdrift_pebbles"].append(v_drift[1].tolist())
            data["St_grains"].append(stokes[0].tolist())
            data["St_pebbles"].append(stokes[1].tolist())

            if planetesimal_params['active']:
                data["Sigma_planetesimals"].append(disc.Sigma_D[2].copy().tolist())
                data["St_planetesimals"].append(stokes[2].tolist())

                data["e_planetesimals"].append(disc._planetesimal.e.copy().tolist())
                data["i_planetesimals"].append(disc._planetesimal.i.copy().tolist())

                if planetesimal_params['drag'] or planetesimal_params['VS_embryo'] or planetesimal_params['VS_pltsml'] or planetesimal_params['DF']:
                    data["de2_dt"].append((disc._planetesimal.de2_dt(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())
                    data["di2_dt"].append((disc._planetesimal.di2_dt(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())

                if planetesimal_params['drag']:
                    data["de2_dt_drag"].append((disc._planetesimal.de2_dt_drag(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())
                    data["di2_dt_drag"].append((disc._planetesimal.di2_dt_drag(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())

                if planetesimal_params['VS_embryo']:
                    data["de2_dt_VS_embryo"].append((disc._planetesimal.de2_dt_VS_embryo(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())
                    data["di2_dt_VS_embryo"].append((disc._planetesimal.di2_dt_VS_embryo(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())

                if planetesimal_params['VS_pltsml']:
                    data["de2_dt_VS_pltsml"].append((disc._planetesimal.de2_dt_VS_pltsml(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())
                    data["di2_dt_VS_pltsml"].append((disc._planetesimal.di2_dt_VS_pltsml(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())

                if planetesimal_params['DF']:
                    data["de2_dt_DF"].append((disc._planetesimal.de2_dt_DF(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())
                    data["di2_dt_DF"].append((disc._planetesimal.di2_dt_DF(disc._planetesimal._e2, disc._planetesimal._i2) * yr).copy().tolist())

            data["T"].append(disc.T.copy().tolist())

            ## Iterate colors
            c1 = next(color1)
            c2 = next(color2)
            c3 = next(color3)
            c4 = next(color4)

            try:
                l, = axes[1].loglog(grid.Rc, disc.Sigma_D[0], linestyle = "dotted", label = 't = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 3)), color = c3)
                axes[1].set_xlabel('Radius [AU]')
                axes[1].set_ylabel('$\\Sigma [g/cm^2]$')
                axes[1].set_ylim(ymin = 1e-6, ymax = 1e5)
                axes[1].set_title('Grain, Pebble, and Gas Surface Density')
                legend1 = axes[1].legend(loc = 'lower left')

            except:
                axes.loglog(grid.Rc, disc.Sigma_G, label = 't = {} yrs'.format(np.round(t / (2 * np.pi))))
                axes.set_xlabel('Radius [AU]')
                axes.set_ylabel('$\\Sigma_{\\mathrm{Gas}} [g/cm^2]$')
                # axes.set_ylim(ymin = 1e-6, ymax = 1e6)
                axes.set_title('Gas and Dust Surface Density')
                axes.legend()

            if transport_params['radial_drift']:
                l2, = axes[1].loglog(grid.Rc, disc.Sigma_D[1], linestyle = "dashdot", color=c2)
                l4, = axes[1].loglog(grid.Rc, disc.Sigma_G, color=c1, linestyle = "dashed")

                if disc._planetesimal:
                    l3, = axes[1].loglog(grid.Rc, disc.Sigma_D[2], color = c4)
                    legend2 = axes[1].legend([l, l2, l3, l4], ["Grains", "Pebbles", "Planetesimals", "Gas"], loc = 'upper right')

                else:
                    legend2 = axes[1].legend([l, l2, l4], ["Grains", "Pebbles", "Gas"], loc = 'upper right')

                axes[1].add_artist(legend1)

                if planet_params['include_planets']:
                    for planet_count, planet_ice_chem in enumerate(X_cores):
                        planetary_mol_abund = SimpleCOMolAbund(len(X_cores[planet_count][0]))
                        planetary_mol_abund.data[:] = (np.array(planet_ice_chem) * np.array(Mcs[planet_count]) + np.array(X_envs[planet_count]) * np.array(Mes[planet_count])) / planets.M[planet_count] # units are not right but doesn't matter if only C/O is being found
                        # planetary_mol_abund.data[:] = np.array(X_envs[count])
                        planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                        planetary_CO = planetary_atom_abund.number_abund("C") / planetary_atom_abund.number_abund("O")
                        planetary_CO = np.nan_to_num(planetary_CO)
                        axes[2].scatter(planets[planet_count].R.copy(), np.array(planets[planet_count].M_core.copy()) + np.array(planets[planet_count].M_env.copy()), color = "black", s = 60, zorder = -1)
                    
            if chemistry_params["on"]:
                atom_abund_ice = disc.chem.ice.atomic_abundance()
                atom_abund_gas = disc.chem.gas.atomic_abundance()

                line1, = axes[3].semilogx(R, atom_abund_ice.number_abund("C") / atom_abund_ice.number_abund("O"), label = f"{t / (2 * np.pi * 10 ** 6):2f} Myr", linestyle = "dashdot", color = c2)
                line2, = axes[3].semilogx(R, atom_abund_gas.number_abund("C") / atom_abund_gas.number_abund("O"), linestyle = "dashed", color = c1)

                if disc._planetesimal:
                    atom_abund_plan = disc._planetesimal.ice_abund.atomic_abundance()
                    line3, = axes[3].semilogx(R, atom_abund_plan.number_abund("C") / atom_abund_plan.number_abund("O"), color = c4)
                    axes[3].legend([line1, line2, line3], ["Grains+Pebbles", "Gas", "Planetesimals"], loc = 'lower right')

                else:
                    axes[3].legend([line1, line2], ["Grains+Pebbles", "Gas"], loc = 'lower right')

                axes[3].set_ylim(0, 1.2)
                axes[3].set_ylabel('[C/O]')
                axes[3].set_xlabel("Radius [AU]")
                axes[3].set_title("C/O ratios throughout the disk")

                d += 1

    if not alpha_SS > 5e-3:
        if planet_params['include_planets']:
            for planet_count, planet_ice_chem in enumerate(X_cores):
                planetary_mol_abund = SimpleCOMolAbund(len(X_cores[planet_count][0]))
                planetary_mol_abund.data[:] = (np.array(planet_ice_chem) * np.array(Mcs[planet_count]) + np.array(X_envs[planet_count]) * np.array(Mes[planet_count])) / planets.M[planet_count] # units are not right but doesn't matter if only C/O is being found
                # planetary_mol_abund.data[:] = np.array(X_envs[planet_count])
                planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                planetary_CO = planetary_atom_abund.number_abund("C") / planetary_atom_abund.number_abund("O")
                planetary_CO = np.nan_to_num(planetary_CO)

                C_O_solar = disc.interp(planets[planet_count].R, X_solar.number_abund("C")) / disc.interp(planets[planet_count].R, X_solar.number_abund("O"))
                axes[0].semilogx(time_keeper, planetary_CO, color = colors[planet_count], label = f"{Rs[planet_count][0]:.0f} AU")

                axes[2].set_prop_cycle(cycler(color = [cm2(planetary_CO[i]) for i in range(len(planetary_CO) - 1)]))

                for i in range(len(planetary_CO) - 1):
                    axes[2].loglog(Rs[planet_count][i:i+2], np.array(Mcs[planet_count][i:i+2]) + np.array(Mes[planet_count][i:i+2]))

        axes[0].set_xlabel("Time (yr)")
        axes[0].legend(loc = "lower right")
        axes[0].set_ylabel("[C/O]")
        axes[0].set_title("C/O of planets over time")

        axes[2].set_xlabel("Radius [AU]")
        axes[2].set_ylabel("Earth Masses")
        axes[2].set_title("Planet Growth Tracks")
        axes[2].set_xlim(1e-1, 500)

        plt.tight_layout()

        sm = plt.cm.ScalarMappable(cmap = cm2)
        cax = fig.add_axes([-0.1, 0, 0.05, 1])
        cbar = fig.colorbar(sm, cax = cax)
        cbar.set_label("C/O Ratio")

        timestamp = time.strftime("%Y%m%d_%H%M")

        filename = (f"{timestamp}_"
                   f"psi{wind_params['psi_DW']:.2g}_"
                   f"Mdot{disc_params['Mdot']:.1e}_"
                   f"M{disc_params['M']:.2g}_"
                   f"Rd{disc_params['Rd']:.0f}")

        fig.savefig(f"{sim_params['figure_dir']}{filename}.png", bbox_inches = 'tight')
        plt.close(fig)

        if planet_params['include_planets']:
            data["Mcs"] = Mcs
            data["Mes"] = Mes
            data["Rp"] = Rs
            data["disk_Mdot_p"] = disk_Mdot_p
            data["t_form"] = (planets.t_form / (2 * np.pi)).tolist() # yr, time each planet was inserted

            if planet_params["planetesimal_accretion_insitu"]:
                data["Mdot_planetesimal"] = Mdot_planetesimal
                data["M_iso_planetesimal"] = M_iso_planetesimal

            if planet_params["pebble_accretion"]:
                data["Mdot_pebble_core"] = Mdot_pebble_core
                data["Mdot_pebble_env"] = Mdot_pebble_env
                data["M_iso_pebble"] = M_iso_pebble

            if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                data["Mdot_migration"] = Mdot_migration

            if planet_params["gas_accretion"]:
                data["Mdot_gas"] = Mdot_gas

            if chemistry_params["on"]:
                data["X_cores"] = X_cores
                data["X_envs"] = X_envs

        data["t"] = time_keeper
        data["disk_Mdot_star"] = disk_Mdot_star
        data["disk_Mass"] = disk_Mass
        data["Tc"] = Tc
        data["Sigc"] = Sigc

        if not wind_params["on"]:
            wind_params["psi_DW"] = 0

        outfile = f"{sim_params['output_dir']}{filename}.h5"

        with h5py.File(outfile, "w") as h5f:
            h5f.attrs["alpha_SS"] = float(alpha_SS)

            for key, value in data.items():
                try:
                    h5f.create_dataset(key, data = value)

                except TypeError:
                    ## Handle ragged lists (e.g. per-planet arrays of different lengths)
                    ## Store as a group of datasets
                    grp = h5f.create_group(key)

                    if isinstance(value, list):
                        for i, arr in enumerate(value):
                            grp.create_dataset(str(i), data = arr)

                    else:
                        grp.attrs["value"] = str(value)

if __name__ == "__main__":
    ## Load config parameters from JSON file
    config_path = "/Users/ben/Downloads/Planet Formation/DiscEvolution Simulations/Config/20260905_full_accretion_psi0.01.json"

    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}", file = sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded config file from: {config_path}")

    ## Run the simulation

    run_model(config)

    print(f"Simulation duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
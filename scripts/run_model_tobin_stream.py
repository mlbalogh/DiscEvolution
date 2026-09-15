## import dependencies
import os
import json
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

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
        # disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], Sigma = Sigma, f_ice = dust_growth_params['f_ice'], thresh = dust_growth_params['thresh'])
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

        ## Planets with Mp = "SI" are only inserted once the planetesimal surface density becomes nonzero
        pending_SI_planets = []

        for i in range(len(Rp)):
            t_impl = planet_params["implant_time"][i]
            R_impl = Rp[i]
            M_impl = Mp[i]

            if str(M_impl).upper() == "SI":
                pending_SI_planets.append((R_impl, M_impl))

            else:
                planet_model.insert_new_planet(t_impl, R_impl, M_impl, planets)

    else:
        planets = None

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

    ## --------------------------
    ## Run model (HDF5 streaming)
    ## --------------------------

    t = 0
    n = 0

    if alpha_SS > 5e-3:
        print ("Not Running model - alpha too high. Alpha, Rd, Mdisk = ", eos.alpha, Rd, disc.Mtot() / Msun)

    else:
        print ("Running model. Alpha, Rd, Mdisk = ", eos.alpha, Rd, disc.Mtot() / Msun)

        if not wind_params["on"]:
            wind_params["psi_DW"] = 0

        timestamp = time.strftime("%Y%m%d_%H%M")
        filename = (f"{timestamp}_"
                   f"psi{wind_params['psi_DW']:.2g}_"
                   f"Mdot{disc_params['Mdot']:.1e}_"
                   f"M{disc_params['M']:.2g}_"
                   f"Rd{disc_params['Rd']:.0f}")
        outfile = f"{sim_params['output_dir']}{filename}.h5"

        with h5py.File(outfile, "w") as h5f:
            ## Scalars
            h5f.create_dataset("t", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.create_dataset("disk_Mdot_star", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.create_dataset("disk_Mass", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.create_dataset("Tc", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.create_dataset("Sigc", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.attrs["alpha_SS"] = float(alpha_SS)

            ## Per-planet extendable datasets
            if planet_params['include_planets']:
                grp_Mcs = h5f.create_group("Mcs")
                grp_Mes = h5f.create_group("Mes")
                grp_Rp = h5f.create_group("Rp")
                grp_Mdotp = h5f.create_group("disk_Mdot_p")

                if planet_params["planetesimal_accretion_insitu"]:
                    grp_Mdot_pltsml = h5f.create_group("Mdot_planetesimal")
                    grp_Miso_pltsml = h5f.create_group("M_iso_planetesimal")

                if planet_params["pebble_accretion"]:
                    grp_Mdot_peb_core = h5f.create_group("Mdot_pebble_core")
                    grp_Mdot_peb_env = h5f.create_group("Mdot_pebble_env")
                    grp_Miso_peb = h5f.create_group("M_iso_pebble")

                if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                    grp_Mdot_mig = h5f.create_group("Mdot_migration")

                if planet_params["gas_accretion"]:
                    grp_Mdot_gas = h5f.create_group("Mdot_gas")

                if chemistry_params["on"]:
                    grp_Xc = h5f.create_group("X_cores")
                    grp_Xe = h5f.create_group("X_envs")

                def create_dataset_backfilled(grp, key, n_backfill):
                    d = grp.create_dataset(str(key), shape = (0,), maxshape = (None,), dtype = "f8", chunks = (1024,))

                    if n_backfill:
                        d.resize(n_backfill, axis = 0)
                        d[:] = np.nan

                    return d

                ## Creates this planet's datasets, backfilling with NaN for any timesteps already recorded
                def create_planet_datasets(ip):
                    n_backfill = h5f["t"].shape[0]

                    for grp in (grp_Mcs, grp_Mes, grp_Rp, grp_Mdotp):
                        create_dataset_backfilled(grp, ip, n_backfill)

                    if planet_params["planetesimal_accretion_insitu"]:
                        for grp in (grp_Mdot_pltsml, grp_Miso_pltsml):
                            create_dataset_backfilled(grp, ip, n_backfill)

                    if planet_params["pebble_accretion"]:
                        for grp in (grp_Mdot_peb_core, grp_Mdot_peb_env, grp_Miso_peb):
                            create_dataset_backfilled(grp, ip, n_backfill)

                    if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                        create_dataset_backfilled(grp_Mdot_mig, ip, n_backfill)

                    if planet_params["gas_accretion"]:
                        create_dataset_backfilled(grp_Mdot_gas, ip, n_backfill)

                    if chemistry_params["on"]:
                        pgrp_c = grp_Xc.create_group(str(ip))
                        pgrp_e = grp_Xe.create_group(str(ip))

                        for js in range(Nchem):
                            create_dataset_backfilled(pgrp_c, js, n_backfill)

                        for js in range(Nchem):
                            create_dataset_backfilled(pgrp_e, js, n_backfill)

                for ip in range(planets.N):
                    create_planet_datasets(ip)

            ## Save the grid once
            h5f.create_dataset("R", data = grid.Rc)

            ## Disk profiles (snapshots)
            nR = len(grid.Rc)
            h5f.create_dataset("time_snap", shape = (0,), maxshape = (None,), dtype = "f8")
            h5f.create_dataset("Sigma_G", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("Sigma_dust", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("Sigma_pebbles", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("Vdrift_grains", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("Vdrift_pebbles", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("St_grains", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("St_pebbles", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
            h5f.create_dataset("T", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

            if planetesimal_params['active']:
                h5f.create_dataset("Sigma_planetesimals", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                h5f.create_dataset("St_planetesimals", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                h5f.create_dataset("e_planetesimals", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                h5f.create_dataset("i_planetesimals", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

                if planetesimal_params['drag'] or planetesimal_params['VS_embryo'] or planetesimal_params['VS_pltsml'] or planetesimal_params['DF']:
                    h5f.create_dataset("de2_dt", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                    h5f.create_dataset("di2_dt", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

                if planetesimal_params['drag']:
                    h5f.create_dataset("de2_dt_drag", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                    h5f.create_dataset("di2_dt_drag", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

                if planetesimal_params['VS_embryo']:
                    h5f.create_dataset("de2_dt_VS_embryo", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                    h5f.create_dataset("di2_dt_VS_embryo", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

                if planetesimal_params['VS_pltsml']:
                    h5f.create_dataset("de2_dt_VS_pltsml", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                    h5f.create_dataset("di2_dt_VS_pltsml", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

                if planetesimal_params['DF']:
                    h5f.create_dataset("de2_dt_DF", shape = (0, nR), maxshape = (None, nR), dtype = "f8")
                    h5f.create_dataset("di2_dt_DF", shape = (0, nR), maxshape = (None, nR), dtype = "f8")

            ## Initial write at t = 0 (if not already included in tinterval)
            disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
            disk_Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU * AU) * (yr / Msun)

            ## Scalars (always written at t = 0 -- the main loop's n % 5 == 0 write
            ## never fires for t = 0 itself, so there is no risk of a duplicate entry
            ## here, unlike the disk-profile snapshots below)
            for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                h5f[name].resize(1, axis = 0)

            h5f["t"][0] = 0.0
            h5f["disk_Mdot_star"][0] = disk_Mdot[0]
            h5f["disk_Mass"][0] = disc.Mtot()
            h5f["Tc"][0] = disc.T[0]
            h5f["Sigc"][0] = disc.Sigma[0]

            ## Per-planet
            if planet_params['include_planets']:
                ## Exact rates at t = 0, before any integrate() call has populated planet_model.rates
                if chemistry_params["on"]:
                    M_Z_0   = (planets.X_core * planets.M_core).sum(0) + (planets.X_env * planets.M_env).sum(0)
                    M_HHe_0 = planets.M_core + planets.M_env - M_Z_0

                else:
                    M_Z_0, M_HHe_0 = planets.M_core, planets.M_env

                initial_rates = planet_model._growth_rates(planets.R, planets.M_core, planets.M_env, M_Z_0, M_HHe_0)

                for ip, planet in enumerate(planets):
                    for name, val, grp in [
                        ("Mcs", planet.M_core.copy(), grp_Mcs),
                        ("Mes", planet.M_env.copy(), grp_Mes),
                        ("Rp", planet.R.copy(), grp_Rp),
                        ("disk_Mdot_p", np.interp(planet.R, grid.Rc[0:-1], disk_Mdot), grp_Mdotp)]:

                        d = grp[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = val

                    if planet_params["planetesimal_accretion_insitu"]:
                        d = grp_Mdot_pltsml[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = initial_rates["Mdot_planetesimal_insitu"][ip] * yr

                        d = grp_Miso_pltsml[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = planet_model._pla_acc.M_iso_pltsml(planet.R)

                    if planet_params["pebble_accretion"]:
                        d = grp_Mdot_peb_core[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = initial_rates["Mdot_pebble_core"][ip] * yr

                        d = grp_Mdot_peb_env[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = initial_rates["Mdot_pebble_env"][ip] * yr

                        d = grp_Miso_peb[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = planet_model._peb_acc.M_iso(planet.R)

                    if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                        d = grp_Mdot_mig[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = initial_rates["Mdot_planetesimal_migration"][ip] * yr

                    if planet_params["gas_accretion"]:
                        d = grp_Mdot_gas[str(ip)]
                        d.resize(1, axis = 0)
                        d[0] = initial_rates["Mdot_gas"][ip] * yr

                    if chemistry_params["on"]:
                        for js, chem in enumerate(planet.X_core):
                            d = grp_Xc[str(ip)][str(js)]
                            d.resize(1, axis = 0)
                            d[0] = chem

                        for js, env in enumerate(planet.X_env):
                            d = grp_Xe[str(ip)][str(js)]
                            d.resize(1, axis = 0)
                            d[0] = env

            ## Disk profiles: gated, since if 0.0 is already one of the requested
            ## snapshot times, the main loop's "for ti in times" will write this
            ## snapshot itself when it reaches ti = 0
            if 0.0 not in sim_params['t_interval']:
                v_drift = disc.v_drift.copy()
                stokes = disc.Stokes().copy()
                for name, arr in [
                    ("Sigma_G", disc.Sigma_G),
                    ("Sigma_dust", disc.Sigma_D[0]),
                    ("Sigma_pebbles", disc.Sigma_D[1]),
                    ("Vdrift_grains", v_drift[0]),
                    ("Vdrift_pebbles", v_drift[1]),
                    ("St_grains", stokes[0]),
                    ("St_pebbles", stokes[1]),
                    ("T", disc.T)]:

                    d = h5f[name]
                    d.resize(1, axis = 0)
                    d[0, :] = arr

                if planetesimal_params['active']:
                    for name, arr in [
                        ("Sigma_planetesimals", disc.Sigma_D[2]),
                        ("St_planetesimals", stokes[2]),
                        ("e_planetesimals", disc._planetesimal.e),
                        ("i_planetesimals", disc._planetesimal.i)]:

                        d = h5f[name]
                        d.resize(1, axis = 0)
                        d[0, :] = arr

                    _e2, _i2 = disc._planetesimal._e2, disc._planetesimal._i2
                    for name, active_flag, fn_e, fn_i in [
                        ("drag", planetesimal_params['drag'], disc._planetesimal.de2_dt_drag, disc._planetesimal.di2_dt_drag),
                        ("VS_embryo", planetesimal_params['VS_embryo'], disc._planetesimal.de2_dt_VS_embryo, disc._planetesimal.di2_dt_VS_embryo),
                        ("VS_pltsml", planetesimal_params['VS_pltsml'], disc._planetesimal.de2_dt_VS_pltsml, disc._planetesimal.di2_dt_VS_pltsml),
                        ("DF", planetesimal_params['DF'], disc._planetesimal.de2_dt_DF, disc._planetesimal.di2_dt_DF)]:

                        if active_flag:
                            d = h5f[f"de2_dt_{name}"]
                            d.resize(1, axis = 0)
                            d[0, :] = fn_e(_e2, _i2) * yr

                            d = h5f[f"di2_dt_{name}"]
                            d.resize(1, axis = 0)
                            d[0, :] = fn_i(_e2, _i2) * yr

                    if planetesimal_params['drag'] or planetesimal_params['VS_embryo'] or planetesimal_params['VS_pltsml'] or planetesimal_params['DF']:
                        d = h5f["de2_dt"]
                        d.resize(1, axis = 0)
                        d[0, :] = disc._planetesimal.de2_dt(_e2, _i2) * yr

                        d = h5f["di2_dt"]
                        d.resize(1, axis = 0)
                        d[0, :] = disc._planetesimal.di2_dt(_e2, _i2) * yr

                h5f["time_snap"].resize(1, axis = 0)
                h5f["time_snap"][0] = 0.0 # Myr

                h5f.flush()

            ## --------------------
            ## Live diagnostic plot
            ## --------------------

            live_plot_enabled = False
            live_update_every = 10 # refresh plot every N timesteps
            fig_live = ax_live1 = ax_live1_r = ax_live2 = None
            gas_line = dust_line = peb_line = plan_line = temp_line = opacity_line = None
            time_label = None

            def update_live_plot():
                if not live_plot_enabled:
                    return
                
                gas_line.set_data(disc.R, disc.Sigma_G)
                dust_line.set_data(disc.R, disc.Sigma_D[0])
                peb_line.set_data(disc.R, disc.Sigma_D[1])

                if plan_line is not None and len(disc.Sigma_D) > 2:
                    plan_line.set_data(disc.R, disc.Sigma_D[2])
                
                temp_line.set_data(disc.R, disc.T)

                if opacity_line is not None and kappa is not None:
                    H = disc.H
                    Sigma_dust = disc.Sigma_D[0]
                    rho_mid = Sigma_dust / (np.sqrt(2 * np.pi) * H)
                    grain_size = disc.grain_size[1]
                    kappa_vals = kappa(rho_mid, disc.T, grain_size)
                    opacity_line.set_data(disc.R, kappa_vals)
                    kappa_finite = kappa_vals[np.isfinite(kappa_vals) & (kappa_vals > 0)]

                    if kappa_finite.size:
                        ax_live1_r.set_ylim(kappa_finite.min() * 0.5, kappa_finite.max() * 2.0)

                time_label.set_text(f"t = {t / (2 * np.pi):.2e} yr")

                def _finite_pos(arr):
                    arr = np.asarray(arr)
                    arr = arr[np.isfinite(arr) & (arr > 0)]
                    return arr if arr.size else np.array([1e-30])

                yvals = np.concatenate([
                    _finite_pos(disc.Sigma_G),
                    _finite_pos(disc.Sigma_D[0]),
                    _finite_pos(disc.Sigma_D[1]),
                    _finite_pos(disc.Sigma_D[2]) if (plan_line is not None and len(disc.Sigma_D) > 2) else np.array([])])
                
                if yvals.size:
                    ymin = max(1e-6, yvals.min() * 0.5)
                    ymax = yvals.max() * 2.0
                    ax_live1.set_ylim(ymin, ymax)

                ax_live1.set_xlim(disc.R.min(), disc.R.max())

                tvals = _finite_pos(disc.T)

                if tvals.size:
                    ax_live2.set_ylim(tvals.min() * 0.9, tvals.max() * 1.1)

                ax_live2.set_xlim(disc.R.min(), disc.R.max())

                fig_live.canvas.draw()
                fig_live.canvas.flush_events()
                plt.pause(0.001)

            if live_plot_enabled:
                plt.ion()
                fig_live, (ax_live1, ax_live2) = plt.subplots(1, 2, figsize = (14, 5))

                gas_line, = ax_live1.loglog(disc.R, disc.Sigma_G, 'k-', linewidth = 2, label = 'Gas')
                dust_line, = ax_live1.loglog(disc.R, disc.Sigma_D[0], 'b--', linewidth = 2, label = 'Dust (grains)')
                peb_line, = ax_live1.loglog(disc.R, disc.Sigma_D[1], 'r:', linewidth = 2, label = 'Pebbles')

                if planetesimal_params['active'] and len(disc.Sigma_D) > 2:
                    plan_line, = ax_live1.loglog(disc.R, disc.Sigma_D[2], 'g-.', linewidth = 2, label = 'Planetesimals')

                ax_live1.set_xlabel('R [AU]', fontsize = 12)
                ax_live1.set_ylabel('Σ [g/cm²]', fontsize = 12)
                ax_live1.set_ylim(1e-3, 5e4)
                ax_live1.set_title('Surface Density (live)', fontsize = 13)
                ax_live1.legend(loc = 'best', fontsize = 10)
                ax_live1.grid(True, alpha = 0.3)
                time_label = ax_live1.text(0.02, 0.95, '', transform = ax_live1.transAxes, ha = 'left', va = 'top')

                ax_live1_r = ax_live1.twinx()

                if kappa is not None:
                    H = disc.H
                    Sigma_dust = disc.Sigma_D[0]
                    rho_mid = Sigma_dust / (np.sqrt(2 * np.pi) * H)
                    grain_size = disc.grain_size[1]
                    kappa_vals = kappa(rho_mid, disc.T, grain_size)
                    opacity_line, = ax_live1_r.loglog(disc.R, kappa_vals, 'm-', linewidth = 2, label = 'κ (opacity)')
                    ax_live1_r.set_ylabel('κ [cm²/g]', fontsize = 12, color = 'm')
                    ax_live1_r.tick_params(axis = 'y', labelcolor = 'm')

                    lines1, labels1 = ax_live1.get_legend_handles_labels()
                    lines2, labels2 = ax_live1_r.get_legend_handles_labels()
                    ax_live1.legend(lines1 + lines2, labels1 + labels2, loc = 'best', fontsize = 10)

                temp_line, = ax_live2.loglog(disc.R, disc.T, 'k-', linewidth = 2)
                ax_live2.set_xlabel('R [AU]', fontsize = 12)
                ax_live2.set_ylabel('T [K]', fontsize = 12)
                ax_live2.set_title('Temperature (live)', fontsize = 13)
                ax_live2.grid(True, alpha = 0.3)

                plt.tight_layout()
                fig_live.canvas.draw()
                fig_live.canvas.flush_events()

            ## ---------------------
            ## Main integration loop
            ## ---------------------

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
                                create_planet_datasets(planets.N - 1)

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

                    if live_plot_enabled and (n % live_update_every == 0):
                        update_live_plot()

                    if (n % 1000) == 0:
                        print('\rNstep: {}'.format(n), flush = "True")
                        print('\rTime: {} Myr'.format(t / (1.e6 * 2 * np.pi)), flush = "True")
                        print('\rdt: {} yr'.format(dt / (2 * np.pi)), flush = "True")

                    ## Stream scalar series every 5 timesteps
                    if n % 5 == 0:
                        k = h5f["t"].shape[0]

                        for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                            h5f[name].resize(k + 1, axis = 0)

                        h5f["t"][k] = t / (2 * np.pi) # years
                        disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
                        disk_Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU * AU) * (yr / Msun)
                        h5f["disk_Mdot_star"][k] = disk_Mdot[0]
                        h5f["disk_Mass"][k] = disc.Mtot()
                        h5f["Tc"][k] = disc.T[0]
                        h5f["Sigc"][k] = disc.Sigma[0]

                    ## Stream per-planet series every 5 timesteps
                    if planet_params['include_planets'] and (n % 5 == 0):
                        for ip, planet in enumerate(planets):
                            for name, val, grp in [
                                ("Mcs", planet.M_core.copy(), grp_Mcs),
                                ("Mes", planet.M_env.copy(), grp_Mes),
                                ("Rp", planet.R.copy(), grp_Rp),
                                ("disk_Mdot_p", np.interp(planet.R, grid.Rc[0:-1], disk_Mdot), grp_Mdotp)]:

                                d = grp[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = val

                            ## planet_model.rates holds the exact rates that drove the most recent integrate() call
                            if planet_params["planetesimal_accretion_insitu"]:
                                d = grp_Mdot_pltsml[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model.rates["Mdot_planetesimal_insitu"][ip] * yr

                                d = grp_Miso_pltsml[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model._pla_acc.M_iso_pltsml(planet.R)

                            if planet_params["pebble_accretion"]:
                                d = grp_Mdot_peb_core[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model.rates["Mdot_pebble_core"][ip] * yr

                                d = grp_Mdot_peb_env[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model.rates["Mdot_pebble_env"][ip] * yr

                                d = grp_Miso_peb[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model._peb_acc.M_iso(planet.R)

                            if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
                                d = grp_Mdot_mig[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model.rates["Mdot_planetesimal_migration"][ip] * yr

                            if planet_params["gas_accretion"]:
                                d = grp_Mdot_gas[str(ip)]
                                d.resize(d.shape[0] + 1, axis = 0)
                                d[-1] = planet_model.rates["Mdot_gas"][ip] * yr

                            if chemistry_params["on"]:
                                for js, chem in enumerate(planet.X_core):
                                    d = grp_Xc[str(ip)][str(js)]
                                    d.resize(d.shape[0] + 1, axis = 0)
                                    d[-1] = chem

                                for js, env in enumerate(planet.X_env):
                                    d = grp_Xe[str(ip)][str(js)]
                                    d.resize(d.shape[0] + 1, axis = 0)
                                    d[-1] = env

                ## Snapshot disk profiles once per timestep
                s = h5f["Sigma_G"].shape[0]
                v_drift = disc.v_drift.copy()
                stokes = disc.Stokes().copy()

                h5f["time_snap"].resize(s + 1, axis = 0);        h5f["time_snap"][s]         = t / (2 * np.pi * 1e6) # Myr
                h5f["Sigma_G"].resize(s + 1, axis = 0);          h5f["Sigma_G"][s, :]        = disc.Sigma_G
                h5f["Sigma_dust"].resize(s + 1, axis = 0);       h5f["Sigma_dust"][s, :]     = disc.Sigma_D[0]
                h5f["Sigma_pebbles"].resize(s + 1, axis = 0);    h5f["Sigma_pebbles"][s, :]  = disc.Sigma_D[1]
                h5f["Vdrift_grains"].resize(s + 1, axis = 0);    h5f["Vdrift_grains"][s, :]  = v_drift[0]
                h5f["Vdrift_pebbles"].resize(s + 1, axis = 0);   h5f["Vdrift_pebbles"][s, :] = v_drift[1]
                h5f["St_grains"].resize(s + 1, axis = 0);        h5f["St_grains"][s, :]      = stokes[0]
                h5f["St_pebbles"].resize(s + 1, axis = 0);       h5f["St_pebbles"][s, :]     = stokes[1]
                h5f["T"].resize(s + 1, axis = 0);                h5f["T"][s, :]              = disc.T

                if planetesimal_params['active']:
                    h5f["Sigma_planetesimals"].resize(s + 1, axis = 0);   h5f["Sigma_planetesimals"][s, :] = disc.Sigma_D[2]
                    h5f["St_planetesimals"].resize(s + 1, axis = 0);      h5f["St_planetesimals"][s, :]    = stokes[2]
                    h5f["e_planetesimals"].resize(s + 1, axis = 0);       h5f["e_planetesimals"][s, :]     = disc._planetesimal.e
                    h5f["i_planetesimals"].resize(s + 1, axis = 0);       h5f["i_planetesimals"][s, :]     = disc._planetesimal.i

                    _e2, _i2 = disc._planetesimal._e2, disc._planetesimal._i2
                    for name, active_flag, fn_e, fn_i in [
                        ("drag", planetesimal_params['drag'], disc._planetesimal.de2_dt_drag, disc._planetesimal.di2_dt_drag),
                        ("VS_embryo", planetesimal_params['VS_embryo'], disc._planetesimal.de2_dt_VS_embryo, disc._planetesimal.di2_dt_VS_embryo),
                        ("VS_pltsml", planetesimal_params['VS_pltsml'], disc._planetesimal.de2_dt_VS_pltsml, disc._planetesimal.di2_dt_VS_pltsml),
                        ("DF", planetesimal_params['DF'], disc._planetesimal.de2_dt_DF, disc._planetesimal.di2_dt_DF)]:

                        if active_flag:
                            h5f[f"de2_dt_{name}"].resize(s + 1, axis = 0); h5f[f"de2_dt_{name}"][s, :] = fn_e(_e2, _i2) * yr
                            h5f[f"di2_dt_{name}"].resize(s + 1, axis = 0); h5f[f"di2_dt_{name}"][s, :] = fn_i(_e2, _i2) * yr

                    if planetesimal_params['drag'] or planetesimal_params['VS_embryo'] or planetesimal_params['VS_pltsml'] or planetesimal_params['DF']:
                        h5f["de2_dt"].resize(s + 1, axis = 0); h5f["de2_dt"][s, :] = disc._planetesimal.de2_dt(_e2, _i2) * yr
                        h5f["di2_dt"].resize(s + 1, axis = 0); h5f["di2_dt"][s, :] = disc._planetesimal.di2_dt(_e2, _i2) * yr

                h5f.flush()

            if planet_params['include_planets']:
                h5f.create_dataset("t_form", data = planets.t_form / (2 * np.pi)) # yr, time each planet was inserted

            ## Mark file complete
            h5f.attrs["complete"] = True

        if live_plot_enabled:
            plt.ioff()
            print("\nLive plot displayed. Close the window to end the program.")



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
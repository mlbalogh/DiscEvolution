import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('Insert/Path/to/DiscEvolution')

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

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
from copy import deepcopy

import time
start_time = time.time()
plt.rcParams.update({'font.size': 16})

#### specify separate temperature 
### initial accreation rate: 2pi r sigma radial velocity at bin 0.

def run_model(config):
    """
    Run the disk evolution model and plot the results.
    
    Parameters:
    config (dict): Configuration dictionary containing all parameters.
    """
    # Extract parameters from config
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
    
    # Set up disc
    # ========================

    # Create the grid
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
    
    # Create the star
    star = SimpleStar(M=star_params["M"], R=star_params["R"], T_eff=star_params['T_eff'])

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

    # define opacity class used. If not Tazzari, defaults to Zhu in IrradiatedEOS.
    if eos_params["opacity"] == "Tazzari":
        kappa = Tazzari2016()
    else:
        kappa = None

    if grid_params['type'] == 'Booth-alpha':
        # For fixed Rd, Mdot and Mdisk, solve for alpha
    
        # extract params
        Mdot=disc_params['Mdot']
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        Rd=disc_params['Rd']
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd/R) * np.exp(-R/Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)
            return Sigma
    
        # define an initial guess for the Sigma profile
        Sigma = Sigma_profile(R, Rd, Mdisk)
    
        # define a gas class, to be used later 
        gas_temp = ViscousEvolutionFV()

        # iterate to get alpha
        for j in range(100):
            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa)
        
            # update eos with current sigma profile
            eos.set_grid(grid)
            eos.update(0, Sigma)

            # define a disc given current eos and Sigma
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find the current Mdot in the disc
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, Sigma))

            # scale Sigma by Mdot to get desired Mdot.
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define new disc given new Sigma profile
            disc = AccretionDisc(grid, star, eos, Sigma)

            # scale alpha by Mdisk so that desired disk mass is achieved.
            alpha= alpha*(disc.Mtot()/Msun)/Mdisk

            if grid_params["smart_bining"]:
                # if using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

    elif grid_params['type'] == 'Booth-Rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd
    
        # extract params
        Mdot=disc_params['Mdot']
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        Rd=disc_params['Rd'] # initial guess
        R = grid.Rc

        def Sigma_profile(R, Rd, Mdisk):
            """Function that creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            Sigma = (Rd/R) * np.exp(-R/Rd)
            Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)
            return Sigma
    
        # create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa)
        
        # update eos with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        # define gas classe to be used in first iteration
        gas_temp = ViscousEvolutionFV()

        # iterate to get Rd
        for j in range(100):
            # initialize a disc with current Sigma and eos
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S=Sigma))

            # Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new= Rd*np.sqrt(Mdisk/(disc.Mtot()/Msun))
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            # define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            # update eos with new Sigma to have correct temperature profile
            eos.update(0, Sigma)

    elif grid_params['type'] == "LBP":
        # define viscous evolution to calculate drift velocity later
        gas = ViscousEvolutionFV()

        # extract parameters
        gamma=disc_params['gamma']
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot']* Msun/yr 
        Mdisk=disc_params['M']* Msun
        alpha=disc_params['alpha']
        mu=chemistry_params['mu']
        rin=R[0]
        xin=R[0]/Rd

        # calculate the keplerian velocity
        fin=np.exp(-xin**(2.-gamma))*(1.-2.*(2.-gamma)*xin**(2.-gamma))
        nud_goal=(Mdot/Mdisk)*(2.*Rd*Rd)/(3.*(2.-gamma))/fin*AU*AU #cm^2
        nud_cgs=nud_goal*yr/3.15e7
        Om_invsecond=star.Omega_k(Rd)*yr/3.15e7

        # calculate initial sound speed and temperature profile
        cs0 = np.sqrt(Om_invsecond*nud_cgs/alpha) #cm/s
        Td=cs0*cs0*mu*m_p/k_B #KT=Td*(R/Rd)**(gamma-1.5)
        T=Td*(R/Rd)**(gamma-1.5)

        # calculate the actual sound speed and surface density profile
        cs = np.sqrt(GasConst * T / mu) #cgs
        cs0 = np.sqrt(GasConst * Td / mu) #cgs
        nu=alpha*cs*cs/(star.Omega_k(R)*yr/3.15e7) # cm2/s
        nud=np.interp(Rd,grid.Rc,nu)*3.15e7/yr # cm^2 
        Sigma=LBP_Solution(Mdisk,Rd*AU,nud,gamma=gamma)
        Sigma0=Sigma(R*AU,0) 

        # Adjust alpha so initial Mdot is correct
        for i in range(10):
            # define an EOS
            eos = IrradiatedEOS(star, alpha_t=disc_params['alpha'], kappa=kappa)
            eos.set_grid(grid)
            eos.update(0, Sigma0)

            # define a temporary disc to compute Mdot
            disc = AccretionDisc(grid, star, eos, Sigma0)

            # adjust alpha depending on current Mdot and wanted Mdot
            vr=gas.viscous_velocity(disc,Sigma0)
            Mdot_actual=disc.Mdot(vr[0])#* (Msun / yr)
            alpha=alpha*(Mdot/Msun*yr)/Mdot_actual
        Sigma = Sigma0

    elif grid_params['type'] == 'Booth-Mdot':
        # For fixed alpha, Rd, and Mdisk, solve for Mdot
    
        # extract parameters
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot']* Msun/yr # initial guess
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        Sigma = (Rd/R) * np.exp(-R/Rd)
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa)
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)

    elif grid_params['type'] == 'winds-alpha':
        # For fixed Rd, Mdot and Mdisk, solve for alpha with disk winds
        # assumes gamma = 1

        # extract params
        Mdot=disc_params['Mdot'] # solar masses per year
        Mdisk=disc_params['M']* Msun
        psi = wind_params['psi_DW']
        #lambda_DW = wind_params['lambda_DW']
        Rd=disc_params['Rd']
        alpha = disc_params['alpha']
        e_rad=wind_params["e_rad"]
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha/(1 + psi)

        # initial guess for Sigma
        Sigma_d = Mdisk/(2 * np.pi * (Rd*AU)**2)
        #xi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
        xi = 0
        Sigma = Sigma_d * (R/Rd)**(xi - gamma) * np.exp(-(R/Rd)**(2 - gamma))

        # define an initial disc and gas class to be used later
        disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
        gas_temp = HybridWindModel(psi, lambda_DW)

        # scale Sigma by current Mtot just in case Sigma is not quite at the correct value to have the desired Mdisk (which often happens)
        Mtot = disc.Mtot()
        Sigma[:] *= Mdisk / Mtot

        for i in range(100):
            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
            
            # update eos with grid and Sigma
            eos.set_grid(grid)
            eos.update(0, Sigma)

            # define new disc 
            disc = AccretionDisc(grid,star,eos,Sigma)

            # find current Mdot in the disc given Sigma and current eos
            vr = gas_temp.viscous_velocity(disc,Sigma)
            Mdot_actual = disc.Mdot(vr)[0] # solar masses per year

            # Scale alpha by Mdot
            alpha_new = alpha*Mdot/Mdot_actual
            alpha = 0.5 * (alpha + alpha_new) # average done to damp oscillations in numerical solution

            # find a new alpha_SS given new alpha.
            alpha_SS = alpha/(1 + psi)

            if grid_params["smart_bining"]:
                # if using smart binning, re-create the grid and Sigma profile
                cutoff = np.where(Sigma < 1e-7)[0]
                
                if cutoff.shape == (0,):
                    continue

                grid_params['rmax'] = grid.Rc[cutoff[0]]
                grid_params['nr'] = cutoff[0]
                grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
                Sigma = np.split(Sigma, [cutoff[0]])[0]

    elif grid_params['type'] == 'winds-Rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd with disk winds
    
        # extract params
        Mdot=disc_params['Mdot'] # solar masses per year
        Mdisk=disc_params['M']* Msun
        psi = wind_params['psi_DW']
        #lambda_DW = wind_params['lambda_DW']
        Rd=disc_params['Rd']
        alpha = disc_params['alpha']
        Sc = disc_params["Sc"]
        gamma = disc_params['gamma']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        R = grid.Rc
        alpha_SS = alpha/(1 + psi)

        def Sigma_profile(R, Rd, Mdisk):
            """Creates a non-steady state Sigma profile for gamma=1, scaled such that the disk mass equals Mdisk"""
            chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
            Sigma = (R/Rd)**(chi - gamma) * np.exp(-(R/Rd)**(2 - gamma))
            Sigma *= Mdisk / np.trapezoid(Sigma, np.pi * (R * AU)**2)
            return Sigma
    
        # create an initial Sigma profile, scale by Mdisk
        Sigma = Sigma_profile(R, Rd, Mdisk)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa)
        
        # update eos with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        # define gas classe to be used in first iteration
        gas_temp = HybridWindModel(psi, lambda_DW)

        # iterate to get Rd
        for j in range(100):
            # initialize a disc with current Sigma and eos
            disc = AccretionDisc(grid, star, eos, Sigma)

            # find Mdot under current parameters
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S=Sigma))

            # Scale Sigma to achieve the desired Mdot
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define a disk with new Sigma profile, use to scale R_d by disk mass
            disc = AccretionDisc(grid, star, eos, Sigma)
            Rd_new= Rd*np.sqrt(Mdisk/disc.Mtot())
            Rd = 0.5 * (Rd + Rd_new) # average done to damp oscillations in numerical solution

            # define new Sigma profile given new Rd
            Sigma = Sigma_profile(R, Rd, Mdisk)

            # update eos with new Sigma to have correct temperature profile
            eos.update(0, Sigma)

    elif grid_params['type'] == 'winds-Mdot':
        # For fixed alpha, Rd, and Mdisk, solve for Mdot with disk winds included
    
        # extract parameters
        R = grid.Rc
        Rd=disc_params['Rd']
        Mdot=disc_params['Mdot'] # initial guess
        Mdisk=disc_params['M']
        alpha=disc_params['alpha']
        psi = wind_params['psi_DW']
        e_rad = wind_params["e_rad"]
        lambda_DW = 1/(2*(1 - e_rad)*(3/psi + 1)) + 1 
        gamma = disc_params['gamma']
        alpha_SS = alpha/(1+psi)

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        chi = 0.25 * (1 + psi) * (np.sqrt(1 + 4*psi/((lambda_DW - 1) * (psi + 1)**2)) - 1)
        Sigma = (R/Rd)**(chi - gamma) * np.exp(-(R/Rd)**(2 - gamma))
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if eos_params["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif eos_params["type"] == "LocallyIsothermalEOS":
            eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha_SS)
        elif eos_params["type"] == "IrradiatedEOS":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)

    # Set up dynamics
    # ========================
    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW)
        else:
            gas = ViscousEvolutionFV()
    
    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        dust = SingleFluidDrift(diffusion=diffuse, settling=dust_growth_params['settling'], van_leer=transport_params['van_leer'])
        diffuse = None

    # Set disc model
    # ========================
    try:
        disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], 
            Sigma=Sigma, feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
            f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
            uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas
        )
    except Exception as e:
        #disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], Sigma=Sigma, f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'])
        raise e

    # Set up Chemistry
    # =======================
    disc.chem = None
    if chemistry_params["on"]:

        # extract params
        N_cell = grid_params["nr"]

        # choose chemical model
        if chemistry_params["chem_model"] == "Simple":
            chemistry = SimpleCOChemOberg()
        elif chemistry_params["chem_model"] == "Equilibrium":
            chemistry = EquilibriumCOChemOberg(a=1e-5)
        elif chemistry_params["chem_model"] == "TimeDep":
            chemistry = TimeDepCOChemOberg(a=1e-5)
        else:
            raise Exception("Valid chemistry model not selected. Choose Simple, Equilibrium, or TimeDep")
        
        # Setup the dust-to-gas ratio from the chemistry
        X_solar = SimpleCOAtomAbund(N_cell) # data array containing abundances of atoms
        X_solar.set_solar_abundances() # redefines abundances by multiplying with specific constants

        # Iterate ice fractions to get the dust-to-gas ratio:
        for i in range(100):
            if chemistry_params["assert_d2g"]:

                # find the total gas and dust mass.
                M_dust = np.trapezoid(disc.Sigma_D.sum(0), np.pi*grid.Rc**2)
                M_gas = np.trapezoid(disc.Sigma_G, np.pi*grid.Rc**2)

                # calculate a modification fraction by dividing the wanted dust fraction by the current dust fraction.
                mod_frac = disc_params["d2g"]/(M_dust/M_gas)

                # multiply modification fraction into dust fraction to assert wanted dust fraction.
                disc.dust_frac[:] = disc.dust_frac*mod_frac

            dust_frac = disc.dust_frac.sum(0)
            
            # Returns MolecularIceAbund class containing SimpleCOMolAbund for gas and ice
            chem = chemistry.equilibrium_chem(disc.T, 
                                            disc.midplane_gas_density,
                                            dust_frac,
                                            X_solar)
            disc.initialize_dust_density(chem.ice.total_abund)
        disc.chem = chem

        disc.update_ices(disc.chem.ice)

    # Set up planetesimals
    # ========================
    disc._planetesimal = None
    if planetesimal_params['active']:
        disc._planetesimal = PlanetesimalFormation(
            disc, 
            d_planetesimal=planetesimal_params['diameter'], 
            St_min=planetesimal_params['St_min'], 
            St_max=planetesimal_params['St_max'], 
            pla_eff=planetesimal_params['pla_eff']
        )

    # Set up planet(s)
    # ========================
    if planet_params['include_planets']:
        
        if chemistry_params["on"]:
            Nchem=disc.chem.ice.data.shape[0]
            planets = Planets(Nchem=Nchem)
        else:
            planets = Planets(Nchem=0)
        
        planet_model = Bitsch2015Model(
            disc, pb_gas_f=planet_params["pb_gas_f"], 
            migrate=planet_params["migrate"], 
            pebble_acc=planet_params["pebble_accretion"],
            gas_acc=planet_params["gas_accretion"],
            planetesimal_acc=planet_params["planetesimal_accretion"],
            winds = wind_params["on"]
        )
        planet_model.set_disc(disc)

        Mp = planet_params['Mp']
        Rp = planet_params['Rp']
        for i in range(len(Mp)):
            if chemistry_params["on"]:
                # we assume planets start with no envelope
                X_core = []

                for d, ice_spec in enumerate(disc.chem.ice.data):
                    X_core.append(disc.interp(Rp[i], ice_spec)/disc.interp(Rp[i], disc.dust_frac[:2].sum(0)))

                X_core = np.array(X_core)
                X_env = np.zeros(X_core.shape)

                planets.add_planet(planet_params["implant_time"][i], Rp[i], Mp[i], 0, X_core, X_env)
            else:
                planets.add_planet(planet_params["implant_time"][i], Rp[i], Mp[i], 0)

        time_keeper = []
        Rs, Mcs, Mes, Mdot_tracker, X_cores, X_envs = [], [], [], [], [], []

        for i, planet in enumerate(planets):
            Rs.append([])
            Mcs.append([])
            Mes.append([])
            Mdot_tracker.append([])

            if chemistry_params["on"]:
                X_cores.append([[] for num in range(0, Nchem, 1)]) 
                X_envs.append([[] for num in range(0, Nchem, 1)])
        
        # first data point
        for count, planet in enumerate(planets):
            Rs[count].append(planet.R.copy())
            Mcs[count].append(planet.M_core.copy())
            Mes[count].append(planet.M_env.copy())
            #Mdot_tracker[count].append(planet_model._peb_acc.computeMdot(planet.R, planet.M))
            if chemistry_params["on"]:
                for count2, chem in enumerate(planet.X_core):
                    X_cores[count][count2].append(chem)
                    X_envs[count][count2].append(planet.X_env[count2])
        time_keeper.append(0)

    # Prepare plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(vr[0])
        
    # display disk characteristics
    plt.figtext(0.5, 0, f"Mdot={Mdot:.3e}, alpha={disc._eos._alpha_t:.3e}, Mtot={disc.Mtot()/Msun:.3e}, Rd={disc.RC():.3e}", ha="center")

    # this is to synchronize colors
    d = 0 
    colors = ["black", "red", "green", "blue", "cyan"]
    cm = plt.get_cmap("viridis")

    # gradient colors also present to give options
    color1=iter(plt.cm.Blues(np.linspace(0.4, 1, 5)[::-1]))
    color2=iter(plt.cm.Greys(np.linspace(0.4, 1, 5)[::-1]))
    color3=iter(plt.cm.Greens(np.linspace(0.4, 1, 5)[::-1]))
    color4=iter(plt.cm.Reds(np.linspace(0.4, 1, 5)[::-1]))

    # Run model
    # ========================
    t = 0
    n = 0

    for ti in times:
        while t < ti:
            # find timestep given gas and dust maximum timesteps
            dt = ti - t
            if transport_params['gas_transport']:
                dt = min(dt, disc._gas.max_timestep(disc))
            if transport_params['radial_drift']:
                dt = min(dt, dust.max_timestep(disc))
            
            # Extract updated dust frac to update gas
            dust_frac = None
            try:
                dust_frac = disc.dust_frac
            except AttributeError:
                pass

            # Extract gas tracers
            gas_chem, ice_chem = None, None
            try:
                gas_chem = disc.chem.gas.data
                ice_chem = disc.chem.ice.data
            except AttributeError:
                pass

            # Do gas evolution
            if transport_params['gas_transport']:
                
                # to preserve planetesimal surface density so 
                # that it doesn't move with a change in Sigma 
                # as a whole, we do the following:
                if disc._planetesimal:
                    disc._gas(dt, disc, [dust_frac[:-1], gas_chem, ice_chem])
                else:
                    disc._gas(dt, disc, [dust_frac, gas_chem, ice_chem])

            # Do dust evolution
            if transport_params['radial_drift']:
                dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)
                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem) #### may use planetesimals to move, double check
                if dust_frac is not None:
                    if disc._planetesimal:
                        # excluding planetesimals (assume they don't move)
                        dust_frac[:2] += dt * diffuse(disc, dust_frac[:2]) 
                    else: 
                        dust_frac[:] += dt * diffuse(disc, dust_frac[:])

            # Pin the values to >= 0 and <=1:
            disc.Sigma[:] = np.maximum(disc.Sigma, 0)     
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
            if chemistry_params["on"]:
                disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

            if disc._planetesimal:
                disc._planetesimal.update(dt, disc, dust)

            if chemistry_params["on"]:
                # exclude planetesimals from chemistry (assume they don't chemically interact with the disc)
                if disc._planetesimal:
                    chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac[:-1].sum(0), disc.chem)
                else:
                    chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac.sum(0), disc.chem)
                
                disc.update_ices(disc.chem.ice)

            if planet_params['include_planets']:
                # Update the planet masses and radii
                planet_model.integrate(dt, planets) 

            # update disc
            disc.update(dt)
            
            # increase time and go forward a steo
            t += dt
            n += 1

            if (n % 1000) == 0:
                print('\rNstep: {}'.format(n), end="", flush="True")
                print('\rTime: {} yr'.format(t / (2 * np.pi)), end="", flush="True")
                print('\rdt: {} yr'.format(dt / (2 * np.pi)), end="", flush="True")
            
            if planet_params['include_planets']:

                # Collect data for planet growth track and chemistry
                if n % 5 == 0:
                    for count, planet in enumerate(planets):
                        Rs[count].append(planet.R.copy())
                        Mcs[count].append(planet.M_core.copy())
                        Mes[count].append(planet.M_env.copy())
                        #Mdot_tracker[count].append(planet_model._peb_acc.computeMdot(planet.R, planet.M))
                        if chemistry_params["on"]:
                            for count2, chem in enumerate(planet.X_core):
                                X_cores[count][count2].append(chem)
                                X_envs[count][count2].append(planet.X_env[count2])
                    time_keeper.append(t/(2*np.pi))

        # iterate colors
        c1 = next(color1)
        c2 = next(color2)
        c3 = next(color3)
        c4 = next(color4)

        try:
            l, = axes[1].loglog(grid.Rc, disc.Sigma_D[0], linestyle="dotted", label='t = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 3)), color=c3)
            axes[1].set_xlabel('Radius [AU]')
            axes[1].set_ylabel('$\\Sigma [g/cm^2]$')
            axes[1].set_ylim(ymin=1e-6, ymax=1e5)
            axes[1].set_title('Grain, Pebble, and Gas Surface Density')
            legend1 = axes[1].legend(loc='lower left')
        except:
            axes.loglog(grid.Rc, disc.Sigma_G, label='t = {} yrs'.format(np.round(t / (2 * np.pi))))
            axes.set_xlabel('Radius [AU]')
            axes.set_ylabel('$\\Sigma_{\\mathrm{Gas}} [g/cm^2]$')
            #axes.set_ylim(ymin=1e-6, ymax=1e6)
            axes.set_title('Gas and Dust Surface Density')
            axes.legend()

        if transport_params['radial_drift']:

            l2, = axes[1].loglog(grid.Rc, disc.Sigma_D[1], linestyle="dashdot", color=c2)
            l4, = axes[1].loglog(grid.Rc, disc.Sigma_G, color=c1, linestyle="dashed")
            if disc._planetesimal:
                l3, = axes[1].loglog(grid.Rc, disc.Sigma_D[2], color=c4)
                legend2 = axes[1].legend([l, l2, l3, l4], ["Grains", "Pebbles", "Planetesimals", "Gas"], loc='upper right')
            else:
                legend2 = axes[1].legend([l, l2, l4], ["Grains", "Pebbles", "Gas"], loc='upper right')
            axes[1].add_artist(legend1)

            for planet_count, planet_ice_chem in enumerate(X_cores):
                planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
                planetary_mol_abund.data[:] = (np.array(planet_ice_chem)*np.array(Mcs[planet_count]) + np.array(X_envs[planet_count])*np.array(Mes[planet_count]))/planets.M[planet_count] ### units are not right but doesn't matter if only C/O is being found.
                #planetary_mol_abund.data[:] = np.array(X_envs[count])
                planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                planetary_CO = planetary_atom_abund.number_abund("C")/planetary_atom_abund.number_abund("O")
                planetary_CO = np.nan_to_num(planetary_CO)
                axes[2].scatter(planets[planet_count].R.copy(), np.array(planets[planet_count].M_core.copy())+np.array(planets[planet_count].M_env.copy()), color="black", s=60, zorder=-1)
            
        if chemistry_params["on"]:
            atom_abund_ice = disc.chem.ice.atomic_abundance()
            atom_abund_gas = disc.chem.gas.atomic_abundance()

            line1, = axes[3].semilogx(R, atom_abund_ice.number_abund("C")/atom_abund_ice.number_abund("O"), label=f"{t/(2*np.pi*10**6):2f} Myr", linestyle="dashdot", color=c2)
            line2, = axes[3].semilogx(R, atom_abund_gas.number_abund("C")/atom_abund_gas.number_abund("O"), linestyle="dashed", color=c1)
            if disc._planetesimal:
                atom_abund_plan = disc._planetesimal.ice_abund.atomic_abundance()
                line3, = axes[3].semilogx(R, atom_abund_plan.number_abund("C")/atom_abund_plan.number_abund("O"), color=c4)
                axes[3].legend([line1, line2, line3], ["Grains+Pebbles", "Gas", "Planetesimals"], loc='lower right')
            else:
                axes[3].legend([line1, line2], ["Grains+Pebbles", "Gas"], loc='lower right')

            axes[3].set_ylim(0, 1.2)
            axes[3].set_ylabel('[C/O]')
            axes[3].set_xlabel("Radius [AU]")
            axes[3].set_title("C/O ratios throughout the disk")

            d+=1

    max_CO = 0
    for planet_count, planet_ice_chem in enumerate(X_cores):
        planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
        planetary_mol_abund.data[:] = (np.array(planet_ice_chem)*np.array(Mcs[planet_count]) + np.array(X_envs[planet_count])*np.array(Mes[planet_count]))/planets.M[planet_count] ### units are not right but doesn't matter if only C/O is being found.
        #planetary_mol_abund.data[:] = np.array(X_envs[planet_count])
        planetary_atom_abund = planetary_mol_abund.atomic_abundance()
        planetary_CO = planetary_atom_abund.number_abund("C")/planetary_atom_abund.number_abund("O")
        planetary_CO = np.nan_to_num(planetary_CO)

        if max_CO < planetary_CO.max():
            max_CO = planetary_CO.max()
        C_O_solar = disc.interp(planets[planet_count].R, X_solar.number_abund("C"))/disc.interp(planets[planet_count].R, X_solar.number_abund("O"))
        axes[0].semilogx(time_keeper, planetary_CO, color=colors[planet_count], label=f"{Rs[planet_count][0]:.0f} AU")

        axes[2].set_prop_cycle(cycler(color=[cm(planetary_CO[i]/max_CO) for i in range(len(planetary_CO)-1)]))
        for i in range(len(planetary_CO)-1):
            axes[2].loglog(Rs[planet_count][i:i+2], np.array(Mcs[planet_count][i:i+2])+np.array(Mes[planet_count][i:i+2]))

    axes[0].set_xlabel("Time (yr)")
    axes[0].legend(loc="lower right")
    axes[0].set_ylabel("[C/O]")
    axes[0].set_title("C/O of planets over time")

    axes[2].set_xlabel("Radius [AU]")
    axes[2].set_ylabel("Earth Masses")
    axes[2].set_title("Planet Growth Tracks")
    axes[2].set_xlim(1e-1, 500)

    plt.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=max_CO))
    cax = fig.add_axes([-0.1, 0, 0.05, 1])
    fig.colorbar(sm, cax=cax)

    fig.savefig('test.png', bbox_inches='tight')

if __name__ == "__main__":
    config = {
        "grid": {
            "rmin": 1e-1,
            "rmax": 1000,
            "nr": 1000,
            "spacing": "natural",
            "smart_bining": False,
            "type": "winds-alpha" # "LBP", "Booth-alpha", "Booth-Rd", "winds-alpha", or "Booth-Mdot"
        },
        "star": {
            "M": 1.0, # Solar masses
            "R": 2.5, # Solar radii
            "T_eff": 4000 # Kelvin
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1e6,
            "t_interval": [0], #[0, 1e-3, 1e-2, 1e-1, 1], Myr
        },
        "disc": {
            "alpha": 1e-3,
            "M": 0.05,
            "d2g": 0.01,
            "Mdot": 1e-8,
            "Sc": 1.0, # schmidt number
            "Rd": 30,
            'gamma': 1
        },
        "eos": {
            "type": "IrradiatedEOS", # "SimpleDiscEOS", "LocallyIsothermalEOS", or "IrradiatedEOS"
            "opacity": "Tazzari",
            "h0": 0.025,
            "q": -0.2
        },
        "transport": {
            "gas_transport": True,
            "radial_drift": True,
            "diffusion": True,
            "van_leer": False
        },
        "dust_growth": {
            "feedback": True,
            "settling": True,
            "f_ice": 1,
            "uf_0": 500,          # Fragmentation velocity for ice-free grains (cm/s)
            "uf_ice": 500,       # Set same as uf_0 to ignore ice effects
            "thresh": 0.5        # Set high threshold to prevent ice effects
        },
        "chemistry": {
            "on"   : True, 
            "fix_mu" : True,
            "mu"     : 2.5,
            "chem_model": "Equilibrium",
            "assert_d2g": True
        },
        "planets": {
            'include_planets': True,
            "planet_model": "Bitsch2015Model",
            "Rp": [1, 5, 10, 20, 30,], #[1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2], #[0.1, 0.1, 0.1, 0.1, 0.1], # initial mass of embryo [M_Earth]
            "implant_time": [0, 0, 0, 0, 0],
            "pb_gas_f": 0.05, # Percent of accreted solids converted to gas
            "migrate" : True,
            "pebble_accretion": True,
            "gas_accretion": True, 
            "planetesimal_accretion": True
        },
        "planetesimal": {
            "active": True,
            "diameter": 200,
            "St_min": 1e-2,
            "St_max": 10,
            "pla_eff": 0.05
        },
        "winds": {
            "on": True,
            "psi_DW": 1,
            "e_rad": 0.9
        }
    }
    run_model(config)
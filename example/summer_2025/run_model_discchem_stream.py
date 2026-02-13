import os
import json
import sys
# Add the path to the DiscEvolution directory
#sys.path.append(os.path.abspath(os.path.join('..')) + '/')
#sys.path.append('Insert/Path/to/DiscEvolution')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler
import h5py

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
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.chemistry import *
from copy import deepcopy

import time
start_time = time.time()
plt.rcParams.update({'font.size': 16})


gas_solver=ViscousEvolutionFV

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
        kappa = Tazzari2016
    elif eos_params["opacity"] == "Zhu2012":
        kappa = Zhu2012
    else:
        kappa = Zhu2012

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
        gas_temp = gas_solver()

        # iterate to get alpha
        for j in range(100):
            # Create the EOS
            if eos_params["type"] == "SimpleDiscEOS":
                eos = SimpleDiscEOS(star, alpha_t=alpha)
            elif eos_params["type"] == "LocallyIsothermalEOS":
                eos = LocallyIsothermalEOS(star, eos_params['h0'], eos_params['q'], alpha)
            elif eos_params["type"] == "IrradiatedEOS":
                eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
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

            if grid_params["smart_binning"]:
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
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
        # update eos with guess Sigma
        eos.set_grid(grid)
        eos.update(0, Sigma)
    
        # define gas classe to be used in first iteration
        gas_temp = gas_solver()

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
        gas = gas_solver()

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
            eos = IrradiatedEOS(star, alpha_t=disc_params['alpha'], kappa=kappa, Tmax=eos_params["Tmax"])
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
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
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
                eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad, Tmax=eos_params["Tmax"])
            
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

            if grid_params["smart_binning"]:
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
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, Tmax=eos_params["Tmax"])
        
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
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad, Tmax=eos_params["Tmax"])
        
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
            gas = gas_solver()
    
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
        Natom=disc.chem.ice.atomic_abundance().data.shape[0] 
        Nmol=disc.chem.gas.data.shape[0]
    else:
        # Set dummy dimensions when chemistry is off
        Natom = 1
        Nmol = 1

    
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
        Rs, Mcs, Mes, Mdot_tracker, X_cores, X_envs, disk_Mdot_star, disk_Mdot_p, disk_Mass, Tc, Sigc = [], [], [], [], [], [], [], [], [], [], []
        

        for i, planet in enumerate(planets):
            Rs.append([])
            Mcs.append([])
            Mes.append([])
            Mdot_tracker.append([])
            disk_Mdot_p.append([])

            if chemistry_params["on"]:
                X_cores.append([[] for num in range(0, Nchem, 1)]) 
                X_envs.append([[] for num in range(0, Nchem, 1)])
        
        # first data point

        disk_v=disc._gas.viscous_velocity(disc, disc.Sigma)
        disk_Mdot=-2*np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU*AU)*(yr/Msun)
        disk_Mdot_star.append(disk_Mdot[0])
        disk_Mass.append(disc.Mtot())
        Tc.append(disc.T[0])
        Sigc.append(disc.Sigma[0])
        for count, planet in enumerate(planets):
            Rs[count].append(planet.R.copy())
            Mcs[count].append(planet.M_core.copy())
            Mes[count].append(planet.M_env.copy())
            #Mdot_tracker[count].append(planet_model._peb_acc.computeMdot(planet.R, planet.M))
            disk_Mdot_p[count].append(np.interp(planet.R,grid.Rc[0:-1],disk_Mdot))
            if chemistry_params["on"]:
                for count2, chem in enumerate(planet.X_core):
                    X_cores[count][count2].append(chem)
                    X_envs[count][count2].append(planet.X_env[count2])

        time_keeper.append(0)

    
    # find Mdot to display below
    vr = disc._gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(vr[0])
        
    # this is to synchronize colors
    d = 0 
    nplanets = len(config["planets"]["Mp"])



    # ========================
    # Run model (HDF5 streaming)
    # ========================
    t = 0
    n = 0
    if (alpha_SS > 0.1) or (alpha_SS<1.e-5):
        print("Not Running model - alphaSS out of range.  Alpha, Rd, Mdisk=", eos._alpha_t, Rd, disc.Mtot()/Msun)
    else:
        print("Running model.  Alpha, Rd, Mdisk=", eos.alpha, Rd, disc.Mtot()/Msun)

        # Output filename (HDF5)
        # Use environment variable DISCEVOLUTION_OUTPUT if set, otherwise fall back to config or default
        output_dir = os.environ.get('DISCEVOLUTION_OUTPUT', sim_params.get('output_dir', './output'))
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"winds_mig_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5"
        outfile = os.path.join(output_dir, filename)

        with h5py.File(outfile, "w") as h5f:

            # Scalars
            h5f.create_dataset("t", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("disk_Mdot_star", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("disk_Mass", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("Tc", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("Sigc", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.attrs["alpha_SS"] = float(alpha_SS)

            # Per-planet extendable datasets

            if planet_params['include_planets']:
                grp_Mcs   = h5f.create_group("Mcs")
                grp_Mes   = h5f.create_group("Mes")
                grp_Rp    = h5f.create_group("Rp")
                grp_Mdotp = h5f.create_group("disk_Mdot_p")
                grp_Xc    = h5f.create_group("X_cores")
                grp_Xe    = h5f.create_group("X_envs")
                nchem_core = len(planets[0].X_core)
                nchem_env  = len(planets[0].X_env)
                for ip in range(nplanets):
                    grp_Mcs.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    grp_Mes.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    grp_Rp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    grp_Mdotp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))

                    pgrp_c = grp_Xc.create_group(str(ip))
                    pgrp_e = grp_Xe.create_group(str(ip))
                    for js in range(nchem_core):
                        pgrp_c.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    for js in range(nchem_env):
                        pgrp_e.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            # Save the grid once
            if "R" not in h5f:
                h5f.create_dataset("R", data=grid.Rc)

            # Disk profiles (snapshots)
            nR = len(grid.Rc)
            h5f.create_dataset("time_snap", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("Sigma_G", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("Sigma_dust", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("Sigma_pebbles", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("Vdrift", shape=(0, 2, nR), maxshape=(None, 2, nR), dtype="f8")
            h5f.create_dataset("Sigma_pebble_size", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("disk_atom_gas_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
            h5f.create_dataset("disk_mol_gas_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")
            h5f.create_dataset("disk_atom_ice_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
            h5f.create_dataset("disk_mol_ice_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")
            if config["planetesimal"]["active"]:
                h5f.create_dataset("Sigma_planetesimals", shape=(0, nR), maxshape=(None, nR), dtype="f8")
                h5f.create_dataset("disk_planetesimal_atom_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
                h5f.create_dataset("disk_planetesimal_mol_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")
            h5f.create_dataset("T", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            
            # ==================================================
            # Initial write at t = 0 (if not already included in tinterval)
            # ==================================================
            disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
            disk_Mdot = -2*np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU*AU)*(yr/Msun)
            if 0.0 not in sim_params['t_interval']:
                # Scalars
                for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                    h5f[name].resize(1, axis=0)
                h5f["t"][0]              = 0.0
                h5f["disk_Mdot_star"][0] = disk_Mdot[0]
                h5f["disk_Mass"][0]      = disc.Mtot()
                h5f["Tc"][0]             = disc.T[0]
                h5f["Sigc"][0]           = disc.Sigma[0]

                # Per-planet

                if planet_params['include_planets']:
                    for ip, planet in enumerate(planets):
                        for name, val, grp in [
                            ("Mcs", planet.M_core.copy(), grp_Mcs),
                            ("Mes", planet.M_env.copy(), grp_Mes),
                            ("Rp", planet.R.copy(), grp_Rp),
                            ("disk_Mdot_p", np.interp(planet.R, grid.Rc[0:-1], disk_Mdot), grp_Mdotp)
                        ]:
                            d = grp[str(ip)]
                            d.resize(1, axis=0)
                            d[0] = val

                        if chemistry_params["on"]:
                            for js, chem in enumerate(planet.X_core):
                                d = grp_Xc[str(ip)][str(js)]
                                d.resize(1, axis=0)
                                d[0] = chem
                            for js, env in enumerate(planet.X_env):
                                d = grp_Xe[str(ip)][str(js)]
                                d.resize(1, axis=0)
                                d[0] = env
                # Disk profiles

                for name, arr in [
                    ("Sigma_G", disc.Sigma_G),
                    ("Sigma_dust", disc.Sigma_D[0]),
                    ("Sigma_pebbles", disc.Sigma_D[1]),
                    ("Sigma_pebble_size", disc.grain_size[1]),
                    ("T", disc.T),
                    ("Vdrift", disc.v_drift),
                    ("disk_atom_gas_abund", disc.chem.gas.atomic_abundance().data if chemistry_params["on"] else np.zeros((Natom, nR))),
                    ("disk_mol_gas_abund", disc.chem.gas.data if chemistry_params["on"] else np.zeros((Nmol, nR))),
                    ("disk_atom_ice_abund", disc.chem.ice.atomic_abundance().data if chemistry_params["on"] else np.zeros((Natom, nR))),
                    ("disk_mol_ice_abund", disc.chem.ice.data if chemistry_params["on"] else np.zeros((Nmol, nR))),
                                    ]:
                    d = h5f[name]
                    d.resize(1, axis=0)
                    d[0, :] = arr
                if config["planetesimal"]["active"]:
                    for name, arr in [
                        ("disk_planetesimal_atom_abund", disc._planetesimal.ice_abund.atomic_abundance().data if chemistry_params["on"] else np.zeros((Natom, nR))),
                        ("disk_planetesimal_mol_abund", disc._planetesimal.ice_abund.data if chemistry_params["on"] else np.zeros((Nmol, nR))),
                        ]:
                        d = h5f[name]
                        d.resize(1, axis=0)
                        d[0, :] = arr
                if config["planetesimal"]["active"]:
                    d = h5f["Sigma_planetesimals"]
                    d.resize(1, axis=0)
                    d[0, :] = disc.Sigma_D[2]
                # Time array
                h5f["time_snap"].resize(1, axis=0)
                h5f["time_snap"][0] = 0.0  # Myr

                h5f.flush()


            # Live diagnostic plot (updates in-place each iteration)
            live_plot_enabled = False
            live_update_every = 10  # refresh plot every N timesteps
            fig_live = ax_live1 = ax_live1_r = ax_live2 = None
            gas_line = dust_line = peb_line = plan_line = temp_line = opacity_line = None
            time_label = None

            def update_live_plot():
                if not live_plot_enabled:
                    return
                # Update line data
                gas_line.set_data(disc.R, disc.Sigma_G)
                dust_line.set_data(disc.R, disc.Sigma_D[0])
                peb_line.set_data(disc.R, disc.Sigma_D[1])
                if plan_line is not None and len(disc.Sigma_D) > 2:
                    plan_line.set_data(disc.R, disc.Sigma_D[2])
                temp_line.set_data(disc.R, disc.T)

                # Compute and update opacity (kappa as function of column density, T, and grain size)
                if opacity_line is not None and kappa is not None:
                    H = disc.H  # scale height
                    Sigma_dust = disc.Sigma_D[0]  # dust surface density
                    rho_mid = Sigma_dust / (np.sqrt(2*np.pi) * H)  # midplane dust density
                    grain_size = disc.grain_size[1]  # pebble grain size
                    kappa_vals = kappa(rho_mid, disc.T, grain_size)
                    opacity_line.set_data(disc.R, kappa_vals)
                    
                    # Update right-hand axis limits for opacity
                    kappa_finite = kappa_vals[np.isfinite(kappa_vals) & (kappa_vals > 0)]
                    if kappa_finite.size:
                        ax_live1_r.set_ylim(kappa_finite.min()*0.5, kappa_finite.max()*2.0)

                # Update time label (years)
                time_label.set_text(f"t = {t/(2*np.pi):.2e} yr")

                # Keep log scales and sensible limits (ignore non-positive values)
                def _finite_pos(arr):
                    arr = np.asarray(arr)
                    arr = arr[np.isfinite(arr) & (arr > 0)]
                    return arr if arr.size else np.array([1e-30])

                yvals = np.concatenate([
                    _finite_pos(disc.Sigma_G),
                    _finite_pos(disc.Sigma_D[0]),
                    _finite_pos(disc.Sigma_D[1]),
                    _finite_pos(disc.Sigma_D[2]) if (plan_line is not None and len(disc.Sigma_D) > 2) else np.array([])
                ])
                if yvals.size:
                    ymin = max(1e-6, yvals.min()*0.5)
                    ymax = yvals.max()*2.0
                    ax_live1.set_ylim(ymin, ymax)
                ax_live1.set_xlim(disc.R.min(), disc.R.max())

                tvals = _finite_pos(disc.T)
                if tvals.size:
                    ax_live2.set_ylim(tvals.min()*0.9, tvals.max()*1.1)
                ax_live2.set_xlim(disc.R.min(), disc.R.max())

                fig_live.canvas.draw()
                fig_live.canvas.flush_events()
                plt.pause(0.001)

            if live_plot_enabled:
                plt.ion()
                fig_live, (ax_live1, ax_live2) = plt.subplots(1, 2, figsize=(14, 5))

                gas_line, = ax_live1.loglog(disc.R, disc.Sigma_G, 'k-', linewidth=2, label='Gas')
                dust_line, = ax_live1.loglog(disc.R, disc.Sigma_D[0], 'b--', linewidth=2, label='Dust (grains)')
                peb_line, = ax_live1.loglog(disc.R, disc.Sigma_D[1], 'r:', linewidth=2, label='Pebbles')
                if config["planetesimal"]["active"] and len(disc.Sigma_D) > 2:
                    plan_line, = ax_live1.loglog(disc.R, disc.Sigma_D[2], 'g-.', linewidth=2, label='Planetesimals')

                ax_live1.set_xlabel('R [AU]', fontsize=12)
                ax_live1.set_ylabel('Σ [g/cm²]', fontsize=12)
                ax_live1.set_ylim(1e-3, 5e4)
                ax_live1.set_title('Surface Density (live)', fontsize=13)
                ax_live1.legend(loc='best', fontsize=10)
                ax_live1.grid(True, alpha=0.3)
                time_label = ax_live1.text(0.02, 0.95, '', transform=ax_live1.transAxes,
                                           ha='left', va='top')

                # Add secondary y-axis for opacity on left panel
                ax_live1_r = ax_live1.twinx()
                if kappa is not None:
                    H = disc.H
                    Sigma_dust = disc.Sigma_D[0]
                    rho_mid = Sigma_dust / (np.sqrt(2*np.pi) * H)
                    grain_size = disc.grain_size[1]
                    kappa_vals = kappa(rho_mid, disc.T, grain_size)
                    opacity_line, = ax_live1_r.loglog(disc.R, kappa_vals, 'm-', linewidth=2, label='κ (opacity)')
                    ax_live1_r.set_ylabel('κ [cm²/g]', fontsize=12, color='m')
                    ax_live1_r.tick_params(axis='y', labelcolor='m')
                    
                    # Add opacity to legend
                    lines1, labels1 = ax_live1.get_legend_handles_labels()
                    lines2, labels2 = ax_live1_r.get_legend_handles_labels()
                    ax_live1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

                temp_line, = ax_live2.loglog(disc.R, disc.T, 'k-', linewidth=2)
                ax_live2.set_xlabel('R [AU]', fontsize=12)
                ax_live2.set_ylabel('T [K]', fontsize=12)
                ax_live2.set_title('Temperature (live)', fontsize=13)
                ax_live2.grid(True, alpha=0.3)

                plt.tight_layout()
                fig_live.canvas.draw()
                fig_live.canvas.flush_events()

            # --------------- Main integration loop ---------------
            # Track wall-clock per step to estimate time to completion
            sim_advanced = 0.0  # total simulated time advanced (code units)
            wall_spent = 0.0    # total wall-clock time spent (seconds)
            # Cache ETA values computed each step to avoid recomputation
            last_eta_seconds = None
            last_eta_hours = 0
            last_eta_minutes = 0
            # Abort simulation if ETA exceeds threshold (hours); configurable via simulation.eta_abort_hours
            eta_abort_hours = sim_params.get('eta_abort_hours', 24)
            early_exit = False  # Flag to track early termination
            dt_min = 1.         # minimum time step; otherwise break early
            for ti in times:
                if early_exit:
                    break
                    
                while t < ti:
                    # Compute physics-limited timestep (for ETA estimation)
                    dt_physics = float('inf')
                    if transport_params['gas_transport']:
                        dt_physics = min(dt_physics, disc._gas.max_timestep(disc))
                    if transport_params['radial_drift']:
                        dt_physics = min(dt_physics, dust.max_timestep(disc))
                    
                    # Actual timestep control (constrained by snapshot time)
                    dt = min(ti - t, dt_physics)

                    # Start wall-clock timer for this iteration
                    step_start = time.time()

                    # dt-based early exit (commented out, preserved for reference)
                    # dt_min_threshold = dt_min * 2 * np.pi  # years in code units
                    # if dt < dt_min_threshold:
                    #     print(f"\n=== WARNING: Timestep too small (dt = {dt/(2*np.pi):.3e} yr) ===", file=sys.stderr)
                    #     print(f"Ending simulation early at t = {t/(2*np.pi*1e6):.6f} Myr", file=sys.stderr)
                    #     print(f"Target time was t = {times[-1]/(2*np.pi*1e6):.6f} Myr", file=sys.stderr)
                    #     early_exit = True
                    #     break

                    # ETA-based early exit: use physics timestep for projection to avoid snapshot-induced artifacts
                    if n >= 1000:
                        remaining_sim = max(times[-1] - t, 0.0)
                        avg_sec_per_step = wall_spent / max(n, 1)
                        # Use physics-limited dt for ETA, not the snapshot-constrained dt
                        dt_for_eta = dt_physics if dt_physics < float('inf') else dt
                        remaining_steps = int(np.ceil(remaining_sim / max(dt_for_eta, 1e-300))) if dt_for_eta > 0 else 0
                        eta_seconds = avg_sec_per_step * remaining_steps
                        # Cache ETA for later printing without recomputation
                        last_eta_seconds = eta_seconds
                        last_eta_hours = int(eta_seconds // 3600)
                        last_eta_minutes = int((eta_seconds % 3600) // 60)
                        if eta_seconds > eta_abort_hours * 3600:
                            print(f"\n=== WARNING: ETA {last_eta_hours:02d}::{last_eta_minutes:02d} h::m exceeds threshold ({eta_abort_hours}h) ===", file=sys.stderr)
                            print(f"Ending simulation early at t = {t/(2*np.pi*1e6):.6f} Myr with dt = {dt/(2*np.pi)} yr", file=sys.stderr)
                            print(f"Target time was t = {times[-1]/(2*np.pi*1e6):.6f} Myr", file=sys.stderr)
                            early_exit = True
                            break

                    dust_frac = getattr(disc, "dust_frac", None)
                    gas_chem, ice_chem = None, None
                    try:
                        gas_chem = disc.chem.gas.data
                        ice_chem = disc.chem.ice.data
                    except AttributeError:
                        pass

                    # gas & dust evolution
                    if transport_params['gas_transport']:
                        if disc._planetesimal:
                            disc._gas(dt, disc, [dust_frac[:-1], gas_chem, ice_chem])
                        else:
                            disc._gas(dt, disc, [dust_frac, gas_chem, ice_chem])
                    if transport_params['radial_drift']:
                        dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

                    if diffuse is not None:
                        if gas_chem is not None:
                            gas_chem[:] += dt * diffuse(disc, gas_chem)
                        if ice_chem is not None:
                            ice_chem[:] += dt * diffuse(disc, ice_chem)
                        if dust_frac is not None:
                            if disc._planetesimal:
                                dust_frac[:2] += dt * diffuse(disc, dust_frac[:2])
                            else:
                                dust_frac[:] += dt * diffuse(disc, dust_frac[:])

                    # bounds
                    disc.Sigma[:] = np.maximum(disc.Sigma, 0)
                    disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
                    disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
                    if chemistry_params["on"]:
                        disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                        disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

                    if disc._planetesimal:
                        disc._planetesimal.update(dt, disc, dust)

                    if chemistry_params["on"]:
                        if disc._planetesimal:
                            chemistry.update(dt, disc.T, disc.midplane_gas_density,
                                            disc.dust_frac[:-1].sum(0), disc.chem)
                        else:
                            chemistry.update(dt, disc.T, disc.midplane_gas_density,
                                            disc.dust_frac.sum(0), disc.chem)
                        disc.update_ices(disc.chem.ice)
                        # These seem to be not used (MLB change Jan 13 2026):
                        # atom_abund_ice = disc.chem.ice.atomic_abundance()
                        # atom_abund_gas = disc.chem.gas.atomic_abundance()
                        
                        # mol_abund_ice = disc.chem.ice
                        # mol_abund_gas = disc.chem.gas
                        # if disc._planetesimal and chemistry_params["on"]:
                        #     atom_abund_planetesimal = disc._planetesimal.ice_abund.atomic_abundance() 
                        #     mol_abund_planetesimal = disc._planetesimal.ice_abund 

                    if planet_params['include_planets']:
                        planet_model.integrate(dt, planets)

                    disc.update(dt)
                    
                    t += dt
                    n += 1

                    # Accumulate wall-clock and simulated time for ETA
                    wall_spent += (time.time() - step_start)
                    sim_advanced += dt

                    # Live plot update every N iterations
                    if live_plot_enabled and (n % live_update_every == 0):
                        update_live_plot()

                    if (n % 1000) == 0:
                        # Use cached ETA from the per-step computation
                        eta_hours = last_eta_hours
                        eta_minutes = last_eta_minutes

                        print(f"\rNstep: {n}", flush=True)
                        print(f"\rTime: {t/(1.e6*2*np.pi)} Myr", flush=True)
                        print(f"\rdt: {dt/(2*np.pi)} yr", flush=True)
                        print(f"\rETA: {eta_hours:02d}::{eta_minutes:02d} (h::m)", flush=True)

                    # --- every 5 steps: stream per-planet series ---
                    if planet_params['include_planets'] and (n % 5 == 0):
                        k = h5f["t"].shape[0]
                        for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                            h5f[name].resize(k + 1, axis=0)

                        h5f["t"][k] = t / (2*np.pi)   # years
                        disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
                        disk_Mdot = -2*np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU*AU) * (yr/Msun)
                        h5f["disk_Mdot_star"][k] = disk_Mdot[0]
                        h5f["disk_Mass"][k] = disc.Mtot()
                        h5f["Tc"][k] = disc.T[0]
                        h5f["Sigc"][k] = disc.Sigma[0]

                        for ip, planet in enumerate(planets):                
                            for name, val, grp in [
                                ("Mcs", planet.M_core.copy(), grp_Mcs),
                                ("Mes", planet.M_env.copy(), grp_Mes),
                                ("Rp", planet.R.copy(), grp_Rp),
                                ("disk_Mdot_p", np.interp(planet.R, grid.Rc[0:-1], disk_Mdot), grp_Mdotp)
                            ]:
                                d = grp[str(ip)]
                                d.resize(d.shape[0] + 1, axis=0)
                                d[-1] = val

                            if chemistry_params["on"]:
                                for js, chem in enumerate(planet.X_core):
                                    d = grp_Xc[str(ip)][str(js)]
                                    d.resize(d.shape[0] + 1, axis=0)
                                    d[-1] = chem
                                for js, env in enumerate(planet.X_env):
                                    d = grp_Xe[str(ip)][str(js)]
                                    d.resize(d.shape[0] + 1, axis=0)
                                    d[-1] = env

                # --- after while loop, once per ti: snapshot disk profiles ---

                s = h5f["Sigma_G"].shape[0]
                h5f["time_snap"].resize(s + 1, axis=0);             h5f["time_snap"][s]      = t / (2*np.pi*1e6)  # Myr
                h5f["Sigma_G"].resize(s + 1, axis=0);               h5f["Sigma_G"][s, :]     = disc.Sigma_G
                h5f["Sigma_dust"].resize(s + 1, axis=0);            h5f["Sigma_dust"][s, :]  = disc.Sigma_D[0]
                h5f["Sigma_pebbles"].resize(s + 1, axis=0);         h5f["Sigma_pebbles"][s,:]= disc.Sigma_D[1]
                h5f["Sigma_pebble_size"].resize(s + 1, axis=0);     h5f["Sigma_pebble_size"][s,:] = disc.grain_size[1]
                if chemistry_params["on"]:
                    h5f["disk_atom_gas_abund"].resize(s + 1, axis=0);  h5f["disk_atom_gas_abund"][s, :, :] = disc.chem.gas.atomic_abundance().data
                    h5f["disk_mol_gas_abund"].resize(s + 1, axis=0);   h5f["disk_mol_gas_abund"][s, :, :]  = disc.chem.gas.data
                    h5f["disk_atom_ice_abund"].resize(s + 1, axis=0);  h5f["disk_atom_ice_abund"][s, :, :] = disc.chem.ice.atomic_abundance().data
                    h5f["disk_mol_ice_abund"].resize(s + 1, axis=0);   h5f["disk_mol_ice_abund"][s, :, :]  = disc.chem.ice.data
                    if config["planetesimal"]["active"]:
                        h5f["disk_planetesimal_atom_abund"].resize(s + 1, axis=0); h5f["disk_planetesimal_atom_abund"][s, :, :] = disc._planetesimal.ice_abund.atomic_abundance().data
                        h5f["disk_planetesimal_mol_abund"].resize(s + 1, axis=0);  h5f["disk_planetesimal_mol_abund"][s, :, :]  = disc._planetesimal.ice_abund.data
                if config["planetesimal"]["active"]:
                    h5f["Sigma_planetesimals"].resize(s + 1, axis=0)
                    h5f["Sigma_planetesimals"][s, :] = disc.Sigma_D[2]
                h5f["T"].resize(s + 1, axis=0);                     h5f["T"][s, :]           = disc.T
                h5f["Vdrift"].resize(s + 1, axis=0);                h5f["Vdrift"][s, :, :]    = disc.v_drift

                h5f.flush()

            # Mark file complete
            h5f.attrs["complete"] = True

        # Turn off interactive mode and keep live plot open
        if live_plot_enabled:
            plt.ioff()
            print("\n=== Live plot displayed. Close the window to end the program. ===")

    # # === Testing: Two-panel final state plot ===
    # print("\n=== Creating final state diagnostic plot ===")
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # # Left panel: Surface densities vs R
    # ax1.loglog(disc.R, disc.Sigma_G, 'k-', linewidth=2, label='Gas')
    # ax1.loglog(disc.R, disc.Sigma_D[0], 'b--', linewidth=2, label='Dust (grains)')
    # ax1.loglog(disc.R, disc.Sigma_D[1], 'r:', linewidth=2, label='Pebbles')
    # if config["planetesimal"]["active"] and len(disc.Sigma_D) > 2:
    #     ax1.loglog(disc.R, disc.Sigma_D[2], 'g-.', linewidth=2, label='Planetesimals')
    # ax1.set_xlabel('R [AU]', fontsize=12)
    # ax1.set_ylabel('Σ [g/cm²]', fontsize=12)
    # ax1.set_ylim(1e-3, 5e4)
    # ax1.set_title('Final Surface Density Profile', fontsize=13)
    # ax1.legend(loc='best', fontsize=10)
    # ax1.grid(True, alpha=0.3)
    
    # # Right panel: Temperature vs R
    # ax2.loglog(disc.R, disc.T, 'k-', linewidth=2)
    # ax2.set_xlabel('R [AU]', fontsize=12)
    # ax2.set_ylabel('T [K]', fontsize=12)
    # ax2.set_title('Final Temperature Profile', fontsize=13)
    # ax2.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # print("Displaying final state plot. Close any plot window to end the program.")
    # plt.show()  # Display and keep open both plots
    # print("Plot closed. Exiting cleanly.")

    
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run disc evolution model with HDF5 streaming output. "
                    "Configuration can be loaded from a JSON file and overridden via command-line arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_model_discchem_stream.py

  # Run with custom config file
  python run_model_discchem_stream.py --config my_config.json

  # Override specific parameters
  python run_model_discchem_stream.py --psi_DW 0.02 --Mdot 5e-9

  # Use environment variable for output directory
  export DISCEVOLUTION_OUTPUT=/path/to/output
  python run_model_discchem_stream.py
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "DiscConfig_default.json"),
        help="Path to configuration JSON file (default: DiscConfig_default.json in script directory)"
    )
    parser.add_argument(
        "--psi_DW",
        type=float,
        default=None,
        help="Override wind parameter psi_DW"
    )
    parser.add_argument(
        "--Mdot",
        type=float,
        default=None,
        help="Override accretion rate [Msun/yr]"
    )
    parser.add_argument(
        "--M",
        type=float,
        default=None,
        help="Override disc mass [Msun]"
    )
    parser.add_argument(
        "--Rd",
        type=float,
        default=None,
        help="Override characteristic disc radius [AU]"
    )
    parser.add_argument(
        "--eta_abort_hours",
        type=float,
        default=None,
        help="Override ETA abort threshold [hours]"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (or use DISCEVOLUTION_OUTPUT environment variable)"
    )
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {args.config}")
    
    # Apply command-line overrides
    if args.psi_DW is not None:
        config["winds"]["psi_DW"] = args.psi_DW
        print(f"Overriding psi_DW: {args.psi_DW}")
    
    if args.Mdot is not None:
        config["disc"]["Mdot"] = args.Mdot
        print(f"Overriding Mdot: {args.Mdot}")
    
    if args.M is not None:
        config["disc"]["M"] = args.M
        print(f"Overriding M: {args.M}")
    
    if args.Rd is not None:
        config["disc"]["Rd"] = args.Rd
        print(f"Overriding Rd: {args.Rd}")
    
    if args.eta_abort_hours is not None:
        config["simulation"]["eta_abort_hours"] = args.eta_abort_hours
        print(f"Overriding eta_abort_hours: {args.eta_abort_hours}")
    
    if args.output_dir is not None:
        config["simulation"]["output_dir"] = args.output_dir
        print(f"Overriding output_dir: {args.output_dir}")
    
    run_model(config)
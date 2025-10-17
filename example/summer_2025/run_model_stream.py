import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('Insert/Path/to/DiscEvolution')

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
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.chemistry import *
from copy import deepcopy

import time
start_time = time.time()
plt.rcParams.update({'font.size': 16})

#### specify separate temperature 
### initial accreation rate: 2pi r sigma radial velocity at bin 0.

def load_visc_data(fname):
    """
    Load viscous disc evolution output file.
    Supports JSON, old HDF5 (single dump / ragged), and new HDF5 (streaming).
    Returns dict in JSON-equivalent format with consistent shapes.
    """

    # ---------------- JSON ----------------
    if fname.endswith(".json"):
        with open(fname, "r") as f:
            data = json.load(f)
    else:
        # ---------------- HDF5 ----------------
        with h5py.File(fname, "r") as f:
            data = {}

            # Scalars
            for key in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                if key in f:
                    data[key] = f[key][()].tolist()
            data["alpha_SS"] = f.attrs.get("alpha_SS", None)

            # Helper for ragged vs extendable vs single-dump
            def read_array(obj):
                if isinstance(obj, h5py.Dataset):       # dataset directly
                    arr = obj[()]
                    if np.ndim(arr) == 0:   # scalar
                        return [float(arr)]
                    return np.array(arr).tolist()
                elif isinstance(obj, h5py.Group):       # ragged group
                    vals = [obj[k][()] for k in sorted(obj.keys(), key=int)]
                    return [float(v) for v in vals]
                else:
                    raise TypeError(f"Unexpected type {type(obj)}")

            # ---------------- Per-planet arrays ----------------
            data["Mcs"], data["Mes"], data["Rp"], data["disk_Mdot_p"] = [], [], [], []

            if isinstance(f.get("Mcs"), h5py.Dataset):
                # Single dataset case
                arr = np.array(f["Mcs"])
                data["Mcs"] = [row.tolist() if arr.ndim > 1 else [float(x)] for row in np.atleast_2d(arr)]
            else:
                for ip in sorted(f["Mcs"].keys(), key=int):
                    data["Mcs"].append(read_array(f["Mcs"][ip]))

            if isinstance(f.get("Mes"), h5py.Dataset):
                arr = np.array(f["Mes"])
                data["Mes"] = [row.tolist() if arr.ndim > 1 else [float(x)] for row in np.atleast_2d(arr)]
            else:
                for ip in sorted(f["Mes"].keys(), key=int):
                    data["Mes"].append(read_array(f["Mes"][ip]))

            if isinstance(f.get("Rp"), h5py.Dataset):
                arr = np.array(f["Rp"])
                data["Rp"] = [row.tolist() if arr.ndim > 1 else [float(x)] for row in np.atleast_2d(arr)]
            else:
                for ip in sorted(f["Rp"].keys(), key=int):
                    data["Rp"].append(read_array(f["Rp"][ip]))

            if isinstance(f.get("disk_Mdot_p"), h5py.Dataset):
                arr = np.array(f["disk_Mdot_p"])
                data["disk_Mdot_p"] = [row.tolist() if arr.ndim > 1 else [float(x)] for row in np.atleast_2d(arr)]
            else:
                for ip in sorted(f["disk_Mdot_p"].keys(), key=int):
                    data["disk_Mdot_p"].append(read_array(f["disk_Mdot_p"][ip]))

            # ---------------- Chemistry ----------------
            data["X_cores"], data["X_envs"] = [], []
            if "X_cores" in f:
                if isinstance(f["X_cores"], h5py.Dataset):
                    # Single dataset case — probably shaped (nplanets, nspecies, nsteps?)
                    arr_core = np.array(f["X_cores"])
                    arr_env  = np.array(f["X_envs"])
                    nplanets = arr_core.shape[0]
                    for ip in range(nplanets):
                        core_species = [arr_core[ip, js, :].tolist() for js in range(arr_core.shape[1])]
                        env_species  = [arr_env[ip, js, :].tolist()  for js in range(arr_env.shape[1])]
                        data["X_cores"].append(core_species)
                        data["X_envs"].append(env_species)
                else:
                    # Group → streaming or ragged format
                    for ip in sorted(f["X_cores"].keys(), key=int):
                        core_species = []
                        env_species  = []
                        for js in sorted(f["X_cores"][ip].keys(), key=int):
                            core_species.append(read_array(f["X_cores"][ip][js]))
                        for js in sorted(f["X_envs"][ip].keys(), key=int):
                            env_species.append(read_array(f["X_envs"][ip][js]))
                        data["X_cores"].append(core_species)
                        data["X_envs"].append(env_species)


            # ---------------- Disk profiles ----------------
            if "R" in f:
                data["R"] = f["R"][:].tolist()
            else:
                nR = f["Sigma_G"].shape[1]
                data["R"] = list(range(nR))  # placeholder if R not saved

            for key in ["Sigma_G", "Sigma_dust", "Sigma_pebbles", "T"]:
                if key in f:
                    data[key] = f[key][:].tolist()
            if "Sigma_planetesimals" in f:
                data["Sigma_planetesimals"] = f["Sigma_planetesimals"][:].tolist()

    # ---------------- Normalize shapes ----------------
    data["R"] = np.array(data["R"], dtype=float).squeeze()  # always (nR,)

    for key in ["Sigma_G", "Sigma_dust", "Sigma_pebbles", "Sigma_planetesimals", "T"]:
        if key in data:
            arr = np.array(data[key], dtype=float)
            arr = np.atleast_2d(arr)          # (nsnapshots, nR)
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]           # squeeze accidental middle dim
            data[key] = arr

    return data


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
            eos = IrradiatedEOS(star, alpha_t=alpha, kappa=kappa, Tmax=eos_params["Tmax"])
        
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
#    colors = ["black", "red", "green", "blue", "cyan"]
    nplanets = len(config["planets"]["Mp"])
    colors = [cm.viridis(i / nplanets) for i in range(nplanets)]

    cm2 = plt.get_cmap("viridis")
    # gradient colors also present to give options
    color1=iter(plt.cm.Blues(np.linspace(0.4, 1, 10)[::-1]))
    color2=iter(plt.cm.Greys(np.linspace(0.4, 1, 10)[::-1]))
    color3=iter(plt.cm.Greens(np.linspace(0.4, 1, 10)[::-1]))
    color4=iter(plt.cm.Reds(np.linspace(0.4, 1, 10)[::-1]))

    # ========================
    # Run model (Option A: JSON-like HDF5 streaming + figure generation)
    # ========================
    t = 0
    n = 0
    if alpha_SS > 5e-3:
        print("Not Running model - alpha too high.  Alpha, Rd, Mdisk=", eos.alpha, Rd, disc.Mtot()/Msun)
    else:
        print("Running model.  Alpha, Rd, Mdisk=", eos.alpha, Rd, disc.Mtot()/Msun)

        # Output filename (HDF5)
        outfile = f"/home/mbalogh/projects/PlanetFormation/DiscEvolution/output/HJpaper/" \
                f"winds_mig_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5"

        with h5py.File(outfile, "w") as h5f:

            # Scalars
            h5f.create_dataset("t", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("disk_Mdot_star", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("disk_Mass", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("Tc", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.create_dataset("Sigc", shape=(0,), maxshape=(None,), dtype="f8")
            h5f.attrs["alpha_SS"] = float(alpha_SS)

            # Per-planet extendable datasets
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
            if config["planetesimal"]["active"]:
                h5f.create_dataset("Sigma_planetesimals", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("T", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            
            # ==================================================
            # Initial write at t = 0
            # ==================================================
            disk_v = disc._gas.viscous_velocity(disc, disc.Sigma)
            disk_Mdot = -2*np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * disk_v * (AU*AU)*(yr/Msun)

            # Scalars
            for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
                h5f[name].resize(1, axis=0)
            h5f["t"][0]              = 0.0
            h5f["disk_Mdot_star"][0] = disk_Mdot[0]
            h5f["disk_Mass"][0]      = disc.Mtot()
            h5f["Tc"][0]             = disc.T[0]
            h5f["Sigc"][0]           = disc.Sigma[0]

            # Per-planet
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
                ("T", disc.T)
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


            # --------------- Main integration loop ---------------
            for ti in times:
                while t < ti:
                    # timestep control
                    dt = ti - t
                    if transport_params['gas_transport']:
                        dt = min(dt, disc._gas.max_timestep(disc))
                    if transport_params['radial_drift']:
                        dt = min(dt, dust.max_timestep(disc))

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

                    if planet_params['include_planets']:
                        planet_model.integrate(dt, planets)

                    disc.update(dt)
                    t += dt
                    n += 1

                    if (n % 1000) == 0:
                        print(f"\rNstep: {n}", flush=True)
                        print(f"\rTime: {t/(1.e6*2*np.pi)} Myr", flush=True)
                        print(f"\rdt: {dt/(2*np.pi)} yr", flush=True)

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
                if config["planetesimal"]["active"]:
                    h5f["Sigma_planetesimals"].resize(s + 1, axis=0)
                    h5f["Sigma_planetesimals"][s, :] = disc.Sigma_D[2]
                h5f["T"].resize(s + 1, axis=0);                     h5f["T"][s, :]           = disc.T

                h5f.flush()

            # Mark file complete
            h5f.attrs["complete"] = True

    
    # ---- Plotting identical to your original code ----
    # if not alpha_SS > 5e-3:
    #     # -------- Figure generation (reuse universal loader) --------
    #     visc_data = load_visc_data(outfile)

    #     # unpack like your notebook expects
    #     time_keeper      = np.array(visc_data["t"])              # years
    #     disk_Mdot_star   = visc_data["disk_Mdot_star"]
    #     disk_Mass        = visc_data["disk_Mass"]
    #     Tc               = visc_data["Tc"]
    #     Sigc             = visc_data["Sigc"]
    #     Mcs              = visc_data["Mcs"]
    #     Mes              = visc_data["Mes"]
    #     Rs               = visc_data["Rp"]
    #     disk_Mdot_p      = visc_data["disk_Mdot_p"]
    #     X_cores          = visc_data["X_cores"]
    #     X_envs           = visc_data["X_envs"]
    #     alpha_SS         = visc_data["alpha_SS"]
    
    #     max_CO = 0
    #     for planet_count, planet_ice_chem in enumerate(X_cores):
    #         planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
    #         planetary_mol_abund.data[:] = (np.array(planet_ice_chem)*np.array(Mcs[planet_count]) +
    #                                        np.array(X_envs[planet_count])*np.array(Mes[planet_count]))/planets.M[planet_count]
    #         planetary_atom_abund = planetary_mol_abund.atomic_abundance()
    #         planetary_CO = planetary_atom_abund.number_abund("C")/planetary_atom_abund.number_abund("O")
    #         planetary_CO = np.nan_to_num(planetary_CO)

    #         if max_CO < planetary_CO.max():
    #             max_CO = planetary_CO.max()
    #         C_O_solar = disc.interp(planets[planet_count].R, X_solar.number_abund("C"))/disc.interp(planets[planet_count].R, X_solar.number_abund("O"))
    #         axes[0].semilogx(time_keeper, planetary_CO, color=colors[planet_count], label=f"{Rs[planet_count][0]:.0f} AU")

    #         axes[2].set_prop_cycle(cycler(color=[cm2(planetary_CO[i]/max_CO) for i in range(len(planetary_CO)-1)]))
    #         for i in range(len(planetary_CO)-1):
    #             axes[2].loglog(Rs[planet_count][i:i+2],
    #                            np.array(Mcs[planet_count][i:i+2])+np.array(Mes[planet_count][i:i+2]))

    #     axes[0].set_xlabel("Time (yr)")
    #     axes[0].legend(loc="lower right")
    #     axes[0].set_ylabel("[C/O]")
    #     axes[0].set_title("C/O of planets over time")

    #     axes[2].set_xlabel("Radius [AU]")
    #     axes[2].set_ylabel("Earth Masses")
    #     axes[2].set_title("Planet Growth Tracks")
    #     axes[2].set_xlim(1e-1, 500)

    #     plt.tight_layout()

    #     sm = plt.cm.ScalarMappable(cmap=cm2, norm=plt.Normalize(vmin=0, vmax=max_CO))
    #     cax = fig.add_axes([-0.1, 0, 0.05, 1])
    #     fig.colorbar(sm, cax=cax)

    #     fig.savefig(f"Figs/winds_mig_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.png",
    #                 bbox_inches='tight')
    #     plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run disc evolution model with optional parameter overrides.")
    parser.add_argument("--psi_DW", type=float, default=None, help="Wind parameter psi_DW")
    parser.add_argument("--Mdot", type=float, default=None, help="Accretion rate [Msun/yr]")
    parser.add_argument("--M", type=float, default=None, help="Disc mass [Msun]")
    parser.add_argument("--Rd", type=float, default=None, help="Characteristic disc radius [AU]")
    args = parser.parse_args()
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
            "t_final": 3.e6,
            "t_interval": [0, 1e-3, 1e-2, 1e-1, 2e-1,5e-1, 1, 2.0, 3.0], #Myr
            #"t_interval": [0, 1e-3, 1e-2, 1e-1,5e-1 ] #Myr
        },
        "disc": {
            "alpha": 1e-3,
            "M": 0.128,
            "d2g": 0.01,
            "Mdot": 8.85e-9, # for Tmax=1500
            #"Mdot": 6.e-8, # with no T cap
            "Sc": 1.0, # schmidt number
            "Rd": 137,
            'gamma': 1
        },
        "eos": {
            "type": "IrradiatedEOS", # "SimpleDiscEOS", "LocallyIsothermalEOS", or "IrradiatedEOS"
            "opacity": "Tazzari",
            "h0": 0.025,
            "q": -0.2,
            "Tmax": 1500.
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
            "Rp": [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30], #[1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], #[0.1, 0.1, 0.1, 0.1, 0.1], # initial mass of embryo [M_Earth]
            "implant_time": [60000, 60000, 60000, 60000, 60000, 600000, 600000, 600000, 600000, 600000, 600000], # 2pi*t(years)
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
            "psi_DW": 0.01,
            "e_rad": 0.9
        }
    }
    # Apply overrides if provided
    if args.psi_DW is not None:
        config["winds"]["psi_DW"] = args.psi_DW
    if args.Mdot is not None:
        config["disc"]["Mdot"] = args.Mdot
    if args.M is not None:
        config["disc"]["M"] = args.M
    if args.Rd is not None:
        config["disc"]["Rd"] = args.Rd
    run_model(config)
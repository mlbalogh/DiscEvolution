import os
import json
import sys
# Add the path to the DiscEvolution directory
sys.path.append(os.path.abspath(os.path.join('..')) + '/')
sys.path.append('/Users/yuvan/GitHub/DiscEvolution/')

import numpy as np
import matplotlib.pyplot as plt

from DiscEvolution.constants import *
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import *
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import *
from DiscEvolution.dust import PlanetesimalFormation
from DiscEvolution.planet_formation import *
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund, SimpleCNOAtomAbund, SimpleCNOMolAbund,
    SimpleCNOChemOberg, TimeDepCNOChemOberg, TimeDepCOChemOberg, EquilibriumCOChemMadhu,
    EquilibriumCNOChemOberg, SimpleCOAtomAbund, SimpleCOChemOberg,
    SimpleCNOChemMadhu, EquilibriumCNOChemMadhu, EquilibriumCOChemOberg
)
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
            Mdot_actual = disc.Mdot(gas_temp.viscous_velocity(disc, S=Sigma))

            # scale Sigma by Mdot to get desired Mdot.
            Sigma_new = Sigma*Mdot/Mdot_actual[0]
            Sigma = 0.5 * (Sigma + Sigma_new) # average done to damp oscillations in numerical solution

            # define new disc given new Sigma profile
            disc = AccretionDisc(grid, star, eos, Sigma)

            # scale alpha by Mdisk so that desired disk mass is achieved.
            alpha= alpha*(disc.Mtot()/Msun)/Mdisk

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
            vr=gas.viscous_velocity(disc,S=Sigma0)
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

    # Set disc model
    # ========================
    try:
        disc = DustGrowthTwoPop(grid, star, eos, disc_params['d2g'], 
            Sigma=Sigma, feedback=dust_growth_params["feedback"], 
            f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
            uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"]
        )
    except Exception as e:
        raise e

    # Set up Chemistry
    # =======================
    if chemistry_params["on"]:
        N_cell = grid_params["nr"]
    
        T =  eos.T
        n = Sigma / (2.4*m_H)

        if chemistry_params["chem_model"] == "Simple":
            chemistry = SimpleCOChemOberg()
        elif chemistry_params["chem_model"] == "Equilibrium":
            chemistry = EquilibriumCOChemOberg(fix_ratios=False, a=1e-5)
        elif chemistry_params["chem_model"] == "TimeDep":
            chemistry = TimeDepCOChemOberg(a=1e-5)
        else:
            raise Exception("Valid chemistry model not selected. Choose Simple, Equilibrium, or TimeDep")
        
        # Setup the dust-to-gas ratio from the chemistry
        X_solar = SimpleCOAtomAbund(N_cell) # data array containing abundances of atoms
        X_solar.set_solar_abundances() # redefines abundances by multiplying with specific constants

        # Iterate ice fractions to get the dust-to-gas ratio:
        for i in range(10):
            if chemistry_params["assert_d2g"]:

                # find the total gas and dust mass
                M_dust = np.trapezoid(disc.Sigma_D.sum(0), np.pi*grid.Rc**2)
                M_gas = np.trapezoid(disc.Sigma_G, np.pi*grid.Rc**2)

                # calculate a modification fraction by dividing the wanted dust fraction by the current dust fraction
                mod_frac = disc_params["d2g"]/(M_dust/M_gas)

                # multiply modification fraction into dust fraction to assert wanted dust fraction.
                disc.dust_frac[:] = disc.dust_frac*mod_frac

            if disc._planetesimal:
                dust_frac = disc.dust_frac[:-1].sum(0)  # exclude planetesimals
            else:
                dust_frac = disc.dust_frac.sum(0)
            
            # Returns MolecularIceAbund class containing SimpleCOMolAbund for gas and ice
            chem = chemistry.equilibrium_chem(disc.T, 
                                            disc.midplane_gas_density,
                                            dust_frac,
                                            X_solar)
            disc.initialize_dust_density(chem.ice.total_abund)
        disc.chem = chem

        #disc.update_ices(disc.chem.ice)

    # Set up dynamics
    # ========================
    gas = None
    if transport_params['gas_transport']:
        #gas = ViscousEvolution(boundary="Mdot_out")
        gas = ViscousEvolution()
    
    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params['Sc'])

    dust = None
    if transport_params['radial_drift']:
        dust = SingleFluidDrift(diffusion=diffuse)
        diffuse = None

    disc._planetesimal = None
    if planetesimal_params['active']:
        disc._planetesimal = PlanetesimalFormation(disc, d_planetesimal=planetesimal_params['diameter'], St_min=planetesimal_params['St_min'], 
                                             St_max=planetesimal_params['St_max'], pla_eff=planetesimal_params['pla_eff'])

    # Set up planet(s)
    # ========================
    if planet_params['include_planets']:
        planets = Planets(Nchem = 0)
        Mp = planet_params['Mp']
        Rp = planet_params['Rp']
        for i in range(len(Mp)):
            planets.add_planet(0, Rp[i], Mp[i], 0)

        planet_model = Bitsch2015Model(disc, pb_gas_f=0.0)
    
    # Prepare plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Set up initial plot lines
    try:
        axes[0].plot(grid.Rc, 0*grid.Rc, '-', color='black')
        axes[0].plot(grid.Rc, 0*grid.Rc, linestyle="dashed", color='black')
        
        # find current Mdot
        vr = gas.viscous_velocity(disc, Sigma=disc.Sigma)
        Mdot = disc.Mdot(vr[0])
        
        # display disc charactersitics
        plt.figtext(0.5, 0, f"Mdot={Mdot:.3e}, alpha={alpha:.3e}, Mtot={disc.Mtot()/Msun:.3e}, Rd={Rd:.3e}", ha="center", va="center")
    except:
        pass
    
    # Figure for planet growth track and planetesimal capture radius
    if planet_params['include_planets']:
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10))
        planet_growth_track = []
        planet_capture_radius = []

    # this is to synchronize colors
    color1=iter(plt.cm.Blues(np.linspace(0.4, 1, 4)[::-1]))
    color2=iter(plt.cm.Greys(np.linspace(0.4, 1, 4)[::-1]))
    color3=iter(plt.cm.Greens(np.linspace(0.4, 1, 4)[::-1]))
    color4=iter(plt.cm.Reds(np.linspace(0.4, 1, 4)[::-1]))

    # Run model
    # ========================
    t = 0
    n = 0

    for ti in times:
        while t < ti:
            # find timestep
            dt = ti - t
            if transport_params['gas_transport']:
                dt = min(dt, gas.max_timestep(disc))
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
                
                # Do the following to preserve planetesimal surface density so that it doesn't move as a tracer
                if disc._planetesimal:
                    gas(dt, disc, [dust_frac[:-1], gas_chem, ice_chem])
                else:
                    gas(dt, disc, [dust_frac, gas_chem, ice_chem])

            # Do dust evolution
            if transport_params['radial_drift']:
                if disc._planetesimal:
                    dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)
                else:
                    dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)
                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem)
                if dust_frac is not None:
                    dust_frac[:-1] += dt * diffuse(disc, dust_frac[:-1]) # excluding planetesimals as we assume they don't move

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
                if disc._planetesimal:
                    # exclude planetesimals, assume they don't significantly contribute to chemistry
                    chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac[:-1].sum(0), disc.chem)
                else:
                    chemistry.update(dt, disc.T, disc.midplane_gas_density, disc.dust_frac.sum(0), disc.chem)

                disc.update_ices(disc.chem.ice)

            if planet_params['include_planets']:
                planet_model.integrate(dt, planets) # Update the planet masses and radii

                # Collect data for planet growth track and capture radius
                for planet in planets:
                    planet_growth_track.append((planet.R, planet.M))
                    planet_capture_radius.append((t / (2 * np.pi * 1e6), planet.R_capt))

            # Do disc update
            disc.update(dt)
            
            # increase time and number of steps
            t += dt
            n += 1

            if (n % 1000) == 0:
                print('\rNstep: {}'.format(n), end="", flush="True")
                print('\rTime: {} yr'.format(t / (2 * np.pi)), end="", flush="True")
                print('\rdt: {} yr'.format(dt / (2 * np.pi)), end="", flush="True")
                
        # iterate color
        c1 = next(color1)
        c2 = next(color2)
        c3 = next(color3)
        c4 = next(color4)

        try:
            l, = axes[1].loglog(grid.Rc, disc.Sigma_D[0]+disc.Sigma_D[1], linestyle="dotted", label='t = {} Myr'.format(np.round(t / (2 * np.pi * 1e6), 3)), color=c3)
            axes[1].set_xlabel('$R\\,[\\mathrm{au}]$')
            axes[1].set_ylabel('$\\Sigma_{\\mathrm{D}} [g/cm^2]$')
            axes[1].set_ylim(ymin=1e-3, ymax=1e5)
            axes[1].set_title('Grain, Pebble, and Planetesimal Surface Density')
            legend1 = axes[1].legend(loc='lower left')
        except:
            axes.loglog(grid.Rc, disc.Sigma_G, label='t = {} yrs'.format(np.round(t / (2 * np.pi))))
            axes.set_xlabel('$R\\,[\\mathrm{au}]$')
            axes.set_ylabel('$\\Sigma_{\\mathrm{Gas}} [g/cm^2]$')
            #axes.set_ylim(ymin=1e-6, ymax=1e6)
            axes.set_title('Gas and Dust Surface Density')
            axes.legend()

        if transport_params['radial_drift'] and not disc._planetesimal:

            #l2, = axes[1].loglog(grid.Rc, disc.Sigma_D[1], linestyle="dashdot", color=c2)
            l4, = axes[1].loglog(grid.Rc, disc.Sigma_G, color=c1, linestyle="dashed")
            #axes[0].set_xlabel('$R\\,[\\mathrm{au}]$')
            #axes[0].set_ylabel('$\\Sigma_{\\mathrm{Planetesimal}} [g/cm^2]$')
            legend2 = axes[1].legend([l, l4], ["Grains+Pebbles", "Gas"], loc='upper right')
            axes[1].add_artist(legend1)

            #axes[3].loglog(grid.Rc, disc.grain_size[1], linestyle="dashed", color=l.get_color())
            #axes[3].set_xlabel('$R\\,[\\mathrm{au}]$')
            #axes[3].set_ylim(ymin=1e-5, ymax=1e1)
            #axes[3].set_ylabel('$a [cm]$')
            #axes[3].set_title('Grain Size')
            #axes[1].set_xlim(0.7, 300)
            #axes[1].set_ylim(10**(-5), 10**(2))

        if planetesimal_params['active']:
            #l2, = axes[1].loglog(grid.Rc, disc.Sigma_D[1], linestyle="dashdot", color=c2)
            l3, = axes[1].loglog(grid.Rc, disc.Sigma_D[2], color=c4, zorder=2)
            l4, = axes[1].loglog(grid.Rc, disc.Sigma_G, color=c1, linestyle="dashed")
            legend2 = axes[1].legend([l, l3, l4], ["Grains+Pebbles", "Planetesimals", "Gas"], loc='upper right')
            axes[1].add_artist(legend1)

            if not chemistry_params["on"]:
                axes[2].loglog(grid.Rc, disc.Stokes()[1], linestyle="dotted", color=l.get_color())
                axes[2].set_xlabel('$R\\,[\\mathrm{au}]$')
                axes[2].set_ylabel('St')
                axes[2].set_title('Stokes Number')
                axes[2].set_ylim(ymin=1e-9, ymax=1e7)
            
        if chemistry_params["on"]:
            atom_abund = disc.chem.ice.atomic_abundance()
            atom_abund_gas = disc.chem.gas.atomic_abundance()

            muH = 1.24
            
            if disc._planetesimal:
                atom_abund_plan = disc._planetesimal.ice_abund.atomic_abundance()
                C_tot = atom_abund.number_abund("C") + atom_abund_plan.number_abund("C")
                O_tot = atom_abund.number_abund("O") + atom_abund_plan.number_abund("O")
                

            #l, = plt.semilogx(R, chem.gas['H2O']/mol_solar['H2O'], '-')
            #plt.semilogx(R, chem.ice['H2O']/mol_solar['H2O'],'--', c=l.get_color())
            p1, = axes[2].semilogx(R, atom_abund.number_abund("C")/atom_abund.number_abund("O"), label=f"{t/10**6:2f} Myr", linestyle="dashed", color=l.get_color())
            if disc._planetesimal:
                p2, = axes[2].semilogx(R, atom_abund_plan.number_abund("C")/atom_abund_plan.number_abund("O"), linestyle="dotted", color=l4.get_color(), zorder=2)
                axes[2].legend([p1, p2], ["Grains+Pebbles", "Planetesimals"], loc='upper right')
            #axes[0].xlim(0.7, 300)
            #axes[0].ylim(0, 1.2)
            axes[2].set_ylabel('[C/O]')
            axes[2].set_title('C/O Ratio of Dust Species')
            axes[2].set_xlabel("Radius (AU)")


            h1, = axes[0].semilogx(R, atom_abund.number_abund("C"), label=f"{t/10**6:2f} Myr", linestyle="dashed", color=l.get_color())
            if disc._planetesimal:
                h2, = axes[0].semilogx(R, atom_abund_plan.number_abund("C"), linestyle="dotted", color=l4.get_color(), zorder=2)
                #h3, = axes[0].semilogx(R, atom_abund_gas.number_abund("C"), linestyle="dotted", color=l2.get_color())
                axes[0].legend([h1, h2], ["Grains+Pebbles", "Planetesimals"], loc='upper right')
            axes[0].set_ylabel("Carbon Fraction") 
            axes[0].set_title('Carbon Fraction of Gas and Dust Species')
            axes[0].set_xlabel("Radius (AU)")
            #axes[2].set_xlim(0.7, 300)
            #axes[2].set_ylim(0, 6)

            a1, = axes[3].semilogx(R, atom_abund.number_abund("O"), label=f"{t/10**6:2f} Myr", linestyle="dashed", color=l.get_color())
            if disc._planetesimal:
                a2, = axes[3].semilogx(R, atom_abund_plan.number_abund("O"), linestyle="dotted", color=l4.get_color(), zorder=2)
                #a3, = axes[3].semilogx(R, atom_abund_gas.number_abund("O"), linestyle="dotted", color=l2.get_color())
                axes[3].legend([a1, a2], ["Grains+Pebbles", "Planetesimals"], loc='upper right')
            axes[3].set_ylabel("Oxygen Fraction") 
            axes[3].set_title('Oxygen fraction of Gas and Dust species')
            axes[3].set_xlabel("Radius (AU)")
            #axes[3].set_ylim(0, 1)

    plt.tight_layout()

    fig.savefig('graphs/chem_w_plan/Danti_repro_2.png', bbox_inches='tight')

if __name__ == "__main__":
    config = {
        "grid": {
            "rmin": 1e-1,
            "rmax": 1000,
            "nr": 1000,
            "spacing": "natural",
            "type": "Booth-Mdot" # "LBP", "Booth-alpha", "Booth-Rd", or "Booth-Mdot"
        },
        "star": {
            "M": 1.0, # Solar masses
            "R": 2.5, # Solar radii
            "T_eff": 4000 # Kelvin
        },
        "simulation": {
            "t_initial": 0,
            "t_final": 1e6,
            "t_interval": [0, 0.5, 1, 3], #[1e-3, 5e-3, 1e-2, 1e-1, 1], # Myr
        },
        "disc": {
            "alpha": 1e-3,
            "M": 0.128,
            "d2g": 0.01, 
            "Mdot": 1e-8,
            "Sc": 1.0, # schmidt number
            "Rd": 137,
            'gamma': 1
        },
        "eos": {
            "type": "IrradiatedEOS",
            "opacity": "Tazzari", # "Tazzari" or "Zhu"
            "h0": 1/30,
            "q": -0.25
        },
        "transport": {
            "gas_transport": True,
            "radial_drift": True,
            "diffusion": True, 
            "van_leer": False
        },
        "dust_growth": {
            "feedback": False,
            "settling": False,
            "f_ice": 0.4,
            "uf_0": 500,          # Fragmentation velocity for ice-free grains (cm/s)
            "uf_ice": 500,        # Set same as uf_0 to ignore ice effects
            "thresh": 0.1        # Set high threshold to prevent ice effects
        },
        "chemistry": {
            "on"   : True, 
            "chem_model": "Equilibrium", 
            "fix_mu" : True,
            "mu"     : 2.5,
            "crate" : 1e-17,
            "assert_d2g": True
        },
        "planets": {
            'include_planets': False,
            "planet_model": "Bitsch2015Model",
            "Rp": [1, 5, 10, 20, 30], # initial position of embryo [AU]
            "Mp": [0.1, 0.1, 0.1, 0.1, 0.1], # initial mass of embryo [M_Earth]
            "migrate" : True
        },
        "planetesimal": {
            "active": True,
            "diameter": 2, # 2
            "St_min": 1e-300,
            "St_max": 1e300,
            "pla_eff": 0.05
        }
    }
    run_model(config)
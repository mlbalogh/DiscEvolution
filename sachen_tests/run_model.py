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
import sys

sys.path.append(os.path.abspath(os.path.join('..')) + '/DiscEvolution')

sys.path.append(os.path.abspath(os.path.join('..')) + '/DiscEvolution/planet_formation_scripts')

import shutil
import json
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import cm, ticker
import matplotlib.pyplot as plt
from DiscEvolution.constants import Msun, AU, yr, Mjup
from DiscEvolution.grid import Grid 
from DiscEvolution.star import SimpleStar, PhotoStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.dust import DustGrowthTwoPop, PlanetesimalFormation, FixedSizeDust
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolution, ViscousEvolutionFV, LBP_Solution, TaboneSolution, HybridWindModel
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
from scipy.special import gamma as gamma_fun
import time

###############################################################################
# Global Constants
###############################################################################

DefaultModel = "sachen_tests/DiscConfig_sachen_tabone.json"

###############################################################################
# Setup Functions
###############################################################################

def setup_disc(model):
    '''Create disc object from initial conditions'''
    # Setup the grid, star and equation of state
    p = model['grid']
    grid = Grid(p['R0'], p['R1'], p['N'], spacing=p['spacing'])

    p = model['star']
    star = SimpleStar(M=p['mass'], R=p['radius'], T_eff=p['T_eff'])
    
    alpha = model['disc']['alpha']
    if alpha == 'standard':
        alpha = np.ones_like(grid.Rc)*1e-3
        alpha[200:300] = 1e-3/100

    p = model['eos']
    if p['type'] == 'irradiated':
        if p['opacity'] == 'Tazzari2016':
            kappa = Tazzari2016()
        elif p['opacity'] == 'Zhu2012':
            kappa = Zhu2012
        else:
            raise ValueError("Opacity not recognised")
        
        eos = IrradiatedEOS(star, alpha, kappa=kappa)
       
    elif p['type'] == 'iso':
        eos = LocallyIsothermalEOS(star, p['h0'], p['q'], 
                                   alpha)
    elif p['type'] == 'simple':
        eos =SimpleDiscEOS(star, alpha)
    else:
        raise ValueError("Error: eos::type not recognised")
    eos.set_grid(grid)
    
    # Setup the physical part of the disc
    p = model['disc']
    if p['type'] == 'Booth-alpha':
        # For fixed Rd, Mdot and Mdisk, solve for alpha
        # did i change this?
        
        Mdot=model['disc']['Mdot']* Msun/yr / AU**2
        Mdisk=model['disc']['mass']* Msun
        Rd=model['disc']['Rc']
        R = grid.Rc
        # Initial guess for Rd and by extension Sigma:
        Sigma = (Mdot / (0.1 * alpha * R**2 * star.Omega_k(R))) * np.exp(-R/Rd)
        eos.update(0, Sigma)
        # iterate to get alpha
        for j in range(10):
            # Iterate to get Mdot
            for i in range(100):
                Sigma = 0.5 * (Sigma + (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd))
                eos.update(0, Sigma)
            Mtot = AccretionDisc(grid, star, eos, Sigma).Mtot()
            alpha=alpha*Mtot/(Mdisk)
            pe = model['eos']
            if pe['type'] == 'irradiated':
                eos =IrradiatedEOS(star, alpha, kappa=kappa)
            elif pe['type'] == 'iso':
                eos = LocallyIsothermalEOS(star, model['eos']['h0'], model['eos']['h0'],alpha)
            elif pe['type'] == 'simple':
                eos =SimpleDiscEOS(star, alpha)
            eos.set_grid(grid)
            eos.update(0, Sigma)
            print (j,alpha,Mtot/Msun)
    elif p['type'] == 'Booth-Rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd
    
        # Initial guess for Sigma:
        Mdot=model['disc']['Mdot']* Msun/yr / AU**2
        Mdisk=model['disc']['mass']* Msun
        Rd=model['disc']['Rc']

        R = grid.Rc

        temporary_arr = np.ndarray((100,3))

        Sigma = (Mdot / (0.1 * alpha * R**2 * star.Omega_k(R))) * np.exp(-R/Rd)
        eos.update(0, Sigma)
        # iterate to get alpha
        for j in range(50):
            # Iterate to get Mdot
            for i in range(100):
                Sigma = 0.5 * (Sigma + (Mdot / (3 * np.pi * eos.nu)) * np.exp(-R/Rd))
                eos.update(0, Sigma)
            Mtot = AccretionDisc(grid, star, eos, Sigma).Mtot()
            Rd=Rd*Mdisk/Mtot
            print (f"{j}, RD: {Rd}, Mtot: {Mtot/Msun}, nu_c: {np.interp(Rd, R, eos.nu)}")
            temporary_arr[j][:] = np.array((Rd,Mtot/Msun,eos.nu[np.abs(R-Rd).argmin()]))
        print ('Rd: ',Rd)
        R = Rd
    elif p['type'] == 'winds-rd':
        # For fixed alpha, Mdot and Mdisk, solve for Rd
    
        # Import variables
        Mdot=model['disc']['Mdot']
        Mdisk=model['disc']['mass']* Msun
        alpha_total=model['disc']['alpha_total']
        Psi = model['disc']['psi']
        Rd=model['disc']['Rc']
        R = grid.Rc
        
        # Create initial surface density
        Lambda = 3
        #Sigma = Mdisk/(2*np.pi*(R*AU)**2)
        xi = 0.25*(1+Psi)*(np.sqrt(1+4*Psi/((Lambda - 1)*(Psi+1)**2))-1)
        # Find SigmaD from M, rc
        SigmaD = Mdisk/(2*np.pi*(Rd*AU)**2* gamma_fun(1 + xi))
        # Find Sigma from SigmaD
        Sigma = SigmaD*(R/Rd)**(xi-1)*np.exp(-R/Rd)

        eos.update(0,Sigma)
        # Create disc
        disc = AccretionDisc(grid, star, eos, Sigma)

        #eos2.update(0,Sigma)
        # iterate to get Rd
        for i in range(50):
            if False:
                disc2 = AccretionDisc(grid, star, eos2, Sigma)
                gas_temporary2 = TaboneSolution(Mdisk/(AU*AU),Rd,eos2.nu[np.abs(R-disc2.RC()).argmin()],Psi)
                #VE_temp = ViscousEvolutionFV()
                #gas_temporary = HybridWindModel(Psi)
                Sigma2 = gas_temporary2(R,0)
                disc2 = AccretionDisc(grid, star, eos2, Sigma)   
                eos2.update(0,Sigma)
       
            # Update sigma based on Rd
            gas_temporary = TaboneSolution(Mdisk/(AU*AU),Rd,1,Psi)
            Sigma = gas_temporary(R, 0)
            
            # Update disc and eos
            eos.update(0,Sigma)
            disc = AccretionDisc(grid, star, eos, Sigma)     
            
            # Find actual Mdot
            vr_visc=gas_temporary.viscous_velocity(disc,Sigma)
            Mdot_actual=disc.Mdot(vr_visc[0])

            print (f"1: Sigma0: {disc.Sigma[0]}, Disc.RC(): {disc.RC()}, Rd: {Rd}, Mdot0: {disc.Mdot(gas_temporary.viscous_velocity(disc))[0]}, Mtotal: {disc.Mtot()/Msun}")

            
            # Scale Sigma by Mdot
            Sigma=Sigma*Mdot/Mdot_actual

            # Update disc and eos
            eos.update(0,Sigma)
            disc = AccretionDisc(grid, star, eos, Sigma)        

            # Scale Rd by Mtot
            Rd = disc.RC()*np.sqrt(Mdisk/(disc.Mtot()))

            print (f"2: Sigma0: {disc.Sigma[0]}, Disc.RC(): {disc.RC()}, Rd: {Rd}, Mdot0: {disc.Mdot(gas_temporary.viscous_velocity(disc))[0]}, Mtotal: {disc.Mtot()/Msun}")


        print ('Rd: ',Rd)
        R = Rd
        alpha_SS = alpha_total/(1+Psi)
        alpha_DW = Psi*alpha_SS
        eos._alpha_DW = alpha_DW
    elif p['type'] == 'LBP':
        gas = ViscousEvolutionFV()
        gamma=model['disc']['gamma']
        R = grid.Rc
        Rd=model['disc']['Rc']
        Mdot=model['disc']['Mdot']* Msun/yr 
        Mdisk=model['disc']['mass']* Msun
        mu=model['eos']['mu']
        rin=R[0]
        xin=R[0]/Rd
        fin=np.exp(-xin**(2.-gamma))*(1.-2.*(2.-gamma)*xin**(2.-gamma))
        nud_goal=(Mdot/Mdisk)*(2.*Rd*Rd)/(3.*(2.-gamma))/fin*AU*AU #cm^2
        nud_cgs=nud_goal*yr/3.15e7
        Om_invsecond=star.Omega_k(Rd)*yr/3.15e7

        cs0 = np.sqrt(Om_invsecond*nud_cgs/alpha) #cm/s
        Td=cs0*cs0*mu*m_p/k_B #KT=Td*(R/Rd)**(gamma-1.5)
        T=Td*(R/Rd)**(gamma-1.5)

        cs = np.sqrt(GasConst * T / mu) #cgs
        cs0 = np.sqrt(GasConst * Td / mu) #cgs
        nu=alpha*cs*cs/(star.Omega_k(R)*yr/3.15e7) # cm2/s
        nud=np.interp(Rd,grid.Rc,nu)*3.15e7/yr # cm^2 
        Sigma=LBP_Solution(Mdisk,Rd*AU,nud,gamma=gamma)
        Sigma0=Sigma(R*AU,0)    
        # Adjust alpha so initial Mdot is correct
        for i in range(10):
            eos = IrradiatedEOS(star, alpha,kappa=kappa)
            #eos = SimpleDiscEOS(star, alpha)
            eos.set_grid(grid)
            eos.update(0,Sigma0)
            disc = AccretionDisc(grid, star, eos, Sigma0)
            vr=gas.viscous_velocity(disc,S=Sigma0)
            Mdot_actual=disc.Mdot(vr[0])#* (Msun / yr)
            alpha=alpha*(Mdot/Msun*yr)/Mdot_actual
            print(alpha, disc.Mtot(), Mdot_actual, eos.nu[0])
        Sigma = Sigma0
    elif p['type'] == 'winds-alpha': 
        
        # Import M, Mdot, rd, psi
        Mdot_disc=model['disc']['Mdot']
        Mdisk=model['disc']['mass']* Msun
        Rd=model['disc']['Rc']
        Psi = model['disc']['psi']
        R = grid.Rc
        Lambda = 3

        # Find kappa for eos
        eos_type = model['eos']

        # Choose temporary alpha
        alpha_total = model['disc']['alpha_total']

        xi = 0.25*(1+Psi)*(np.sqrt(1+4*Psi/((Lambda - 1)*(Psi+1)**2))-1)
        # Find SigmaD from M, rc
        SigmaD = Mdisk/(2*np.pi*(Rd*AU)**2* gamma_fun(1 + xi))
        # Find Sigma from SigmaD
        Sigma = SigmaD*(R/Rd)**(xi-1)*np.exp(-R/Rd)

        # Find eos with sigma, alpha (gives cs)
        eos.update(0,Sigma)
        print(f"cs0: {eos.cs[0]}, alpha_total: {alpha_total}")
        # Iterate finding alpha from cs and cs from alpha
        for i in range(100):
              
            disc = AccretionDisc(grid,star,eos,Sigma)
            #gas_temp = HybridWindModel(Psi, 3)
            gas_temp = TaboneSolution(Mdisk/(AU*AU),Rd,1,Psi,model['disc']['d2g'])

            vr = gas_temp.viscous_velocity(disc,Sigma)
            Mdot = disc.Mdot(vr)[0]
            
            # Scale alpha_SS by Mdot
            alpha_total = alpha_total*Mdot_disc/Mdot
            alpha_SS = alpha_total/(1+Psi)
            
            # Find new eos with new alpha and sigma
            if eos_type['type'] == 'irradiated':
                eos = IrradiatedEOS(eos._star, alpha_SS, kappa=eos._kappa)
            elif eos_type['type'] == 'iso':
                eos = LocallyIsothermalEOS(star, eos._h0, eos._q, alpha_SS)
            eos.set_grid(grid)
            eos.update(0,Sigma)

            print(f"Mtot: {disc.Mtot()/Msun}, cs0: {eos.cs[0]}, alpha_t: {alpha_total}, Mdot: {Mdot}")
        
        alpha_DW = Psi*alpha_SS
    elif p['type'] == 'winds-mdot-var-alpha': 
        # For fixed alpha, Rd, and Mdisk, solve for Mdot

        # extract parameters
        R = grid.Rc
        Rd=p['Rc']
        Mdot=p['Mdot'] # initial guess
        Mdisk=p['mass']
        
        alpha_SS = np.ones_like(R)*1e-3
        alpha_SS[11:20] = 1e-5
        alpha_DW = np.ones_like(R)*1e-3
        Psi = alpha_DW/alpha_SS

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        Sigma = (Rd/R) * np.exp(-R/Rd)
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if model['eos']["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif model['eos']["type"] == "iso":
            eos = LocallyIsothermalEOS(star, model['eos']['h0'], model['eos']['q'], alpha_SS)
        elif model['eos']["type"] == "irradiated":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa)
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)
        disc = AccretionDisc(grid,star,eos,Sigma)
        v_visc = HybridWindModel(Psi).viscous_velocity(disc)
        print(f"alpha: {alpha}, Alpha_SS: {alpha_SS}, Mtot: {disc.Mtot()}, Mdot: {disc.Mdot(v_visc)[0]}, Rd: {disc.RC()}")
    elif p['type'] == 'winds-mdot': 
        # For fixed alpha, Rd, and Mdisk, solve for Mdot

        # extract parameters
        R = grid.Rc
        Rd=p['Rc']
        Mdot=p['Mdot'] # initial guess
        Mdisk=p['mass']
        Psi = p['psi']
        if Psi == 'standard':
            Psi = np.ones_like(R)
            Psi[11:20] = 100
        alpha_total = model['disc']['alpha_total']
        alpha_SS = alpha_total/(1+Psi)
        alpha_DW = Psi * alpha_SS

        # define Sigma profile, scale by Mdisk to get correct disk mass.
        Sigma = (Rd/R) * np.exp(-R/Rd)
        Sigma *= Mdisk / (np.trapezoid(Sigma, np.pi * (R * AU)**2)/Msun)

        # Create the EOS
        if model['eos']["type"] == "SimpleDiscEOS":
            eos = SimpleDiscEOS(star, alpha_t=alpha_SS)
        elif model['eos']["type"] == "iso":
            eos = LocallyIsothermalEOS(star, model['eos']['h0'], model['eos']['q'], alpha_SS)
        elif model['eos']["type"] == "irradiated":
            eos = IrradiatedEOS(star, alpha_t=alpha_SS, kappa=kappa)
        
        # update the eos with relevant values
        eos.set_grid(grid)
        eos.update(0, Sigma)
        disc = AccretionDisc(grid,star,eos,Sigma)
        
        #v_visc3 = ViscousEvolutionFV().viscous_velocity(disc)
        #v_visc2 = HybridWindModel(Psi).viscous_velocity(disc)
        v_visc = TaboneSolution(disc.Mtot()/AU**2,disc.RC(),1,Psi,p['d2g']).viscous_velocity(disc)
        print(f"alpha: {alpha}, Alpha_SS: {alpha_SS}, Mtot: {disc.Mtot()}, Mdot: {disc.Mdot(v_visc)[0]}, Rd: {disc.RC()}")
    else:
        Sigma = np.exp(-grid.Rc / p['Rc']) / (grid.Rc)
        Sigma *= p['mass'] / np.trapz(Sigma, np.pi*grid.Rc**2)
        Sigma *= Msun / AU**2
  
    eos.update(0, Sigma)
    try:
        feedback = model['disc']['feedback']
    except KeyError:
        feedback = True
    
    if model['disc']['d2g'] > 0:
        amin = model['disc']['amin']
        if model['transport']['type'] == "Tabone-Analytic":
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, 
                                amin=amin, Sc=1, 
                                f_grow=model['disc'].get('f_grow',1.0),
                                feedback=feedback,rho_s = model['dust']['density'])
        else:
            disc = DustGrowthTwoPop(grid, star, eos, p['d2g'], Sigma=Sigma, 
                                amin=amin, Sc=model['disc']['Schmidt'], 
                                f_grow=model['disc'].get('f_grow',1.0),
                                feedback=feedback,rho_s = model['dust']['density'])
    else:
        disc = AccretionDisc(grid, star, eos, Sigma)
    #Mdot_calc = 3*np.pi/Omega_k_in*(2/reduce_array(R)*(np.diff(Sigma*eos.cs**2*alpha_SS*R**2)/grid.dRc)+reduce_array((Sigma*eos.cs**2*alpha_SS*Psi)))/(Msun/yr / AU**2)
    try :
        print(f"Disc mass: {disc.Mtot()/Msun}, Mdot (not from   v_visc): {Mdot}, Rd: {disc.RC()}, Psi: {disc._eos._alpha_DW/disc.alpha}")
    except :
        print(f"Disc mass: {disc.Mtot()/Msun}, Mdot (not from v_visc): {Mdot }, Rd: {disc.RC()}, alpha: {disc.alpha}")
    
    return disc


def setup_model(model, disc, history, start_time=0, internal_photo_type="Primordial", R_hole=None):
    '''Setup the physics of the model'''
    
    gas       = None
    dust      = None
    diffuse   = None
    chemistry = None
    planetesimal = None
    planets = None
    planet_model = None

    if model['transport']['gas']:
        if model['transport']['type'] == "Tabone-Analytic":
            psi = model['disc']['psi']
            if psi == 'standard':
                psi = np.ones_like(disc.R)
                psi[11:20] = 100
            gas = TaboneSolution(disc.Mtot()/AU**2, disc.RC(), np.interp(disc.RC(), disc.grid.Rc, disc._eos.nu),psi, d2g=model['disc']['d2g'])# Wrong? #(disc.Mdot(v_visc)/3)*(2*disc.RC()**2/disc.Mtot()),psi)
            t_acc = gas._tc
        if model['transport']['type'] == "HybridWind":
            psi = model['disc']['psi']
            if psi == 'standard':
                psi = np.ones_like(disc.R)
                psi[11:20] = 100
            gas = HybridWindModel(psi, boundary = 'Zero')# Wrong? #(disc.Mdot(v_visc)/3)*(2*disc.RC()**2/disc.Mtot()),psi)
        elif model['transport']['type'] == "ViscousEvolution":
            try:
                gas = ViscousEvolution(boundary=model['grid']['outer_bound'], in_bound=model['grid']['inner_bound'])
            except KeyError:
                print("Default boundaries")
                gas = ViscousEvolution(boundary='Mdot_out')
        else:
            raise Exception
    disc.set_gas(gas)
    if model['transport']['diffusion']:
        diffuse = TracerDiffusion(Sc=model['disc']['Schmidt'])
    if model['transport']['radial drift']:
        dust = SingleFluidDrift(diffuse)
        diffuse = None

    if model['planetesimal']['active']:
        planetesimal_params = model['planetesimal']
        planetesimal = PlanetesimalFormation(disc, d_planetesimal=planetesimal_params['diameter'], St_min=planetesimal_params['St_min'], 
                                             St_max=planetesimal_params['St_max'], pla_eff=planetesimal_params['pla_eff'])
        disc._planetesimal = planetesimal
    if model['planets']['active']:
        planet_params = model['planets']
        
        planets = Planets(Nchem = 0)

        for i in range(len(planet_params['planets'])):
            planets.add_planet(0, planet_params['planets'][i]['Rp'], planet_params['planets'][i]['Mp'], 0)

        if planet_params['planet_model'] == 'Bitsch2015Model':
            planet_model = Bitsch2015Model(disc, pb_gas_f=0.0,migrate=planet_params['migrate'], planetesimal_acc = model['planets']['planetesimal_accretion'], pebble_acc=model['planets']['pebble_accretion'])
            planet_model.set_disc(disc)

        #planet_model = photorate.PlanetModel(planets['mass'], planets['radius'], planets['a'], planets['ecc'], planets['inc'], planets['f'], planets['t0'])
    
    if isinstance(gas,TaboneSolution) and False:
        #gas._nuc = 9.948958718676753e-06
        xs = disc.R
        #ts = np.array([1e3,1e4,1e5,1e6])*2*np.pi
        #ts = np.array([0,2,4,6,8])*t_acc
        ts = np.array([0, 2e5, 4e5, 6e5, 8e5, 1e6])*yr
        ys = []
        for t in ts:
            ys.append(gas(xs,t))
        matplotlib.use('TkAgg') 
        fig, ax = plt.subplots()
        for i in range(len(ts)):
            ax.loglog(xs,ys[i])
        ax.set_ylim((10e-4,10e4))
        ax.set_title(f"Psi {psi}")
        plt.show()

    return PlanetDiscDriver(disc, gas=gas, dust=dust, planetesimal=planetesimal, diffusion=diffuse, planets=planets, planet_model=planet_model,
                            ext_photoevaporation=None, int_photoevaporation=None, history=history, t0=start_time)


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
    if out['save_times'] is not None:
        output_times = np.array(out['save_times'])*yr
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

# Write planet information
def save_planets(model, filename, planets, time):
    if model.num_steps == 0:
        mode = 'w'
    else:
        mode = 'a'
    with open(filename,mode) as f:
        if mode == 'w':
            f.write('Time Planet_Number Mp R Menv Mdot\n')
        for idx in range(planets.N):
            f.write(f"{time} {idx} {planets.M_core[idx]} {planets.R[idx]} {planets.M_env[idx]} {planets.Mdot[idx]}\n")
                
def reduce_array(arr):
    return 0.5*(arr[1:] + arr[:-1])

def drazkowska_plot(pl_acc, location, t):
    Rp = np.logspace(np.log10(0.3),np.log10(2e2),50)
    Mp = np.logspace(np.log10(9e-9),np.log10(2e2),50)

    X, Y = np.meshgrid(Rp,Mp)

    tau = np.empty_like(X)#= pl_acc.computeMdotTwoPhase(X,Y,None)[0]
 
    for i in range(len(Y)):
        if i == 15:
            print("")
        Mdot= pl_acc.computeMdot(X[i],Y[i],pl_acc.dRdt)
        tau[i] = Y[i]/(Mdot*(2*np.pi))

    oli_run = pl_acc.m_olig_addition(X,Y)

    def fmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    colors = cm.Greens_r(np.linspace(0.2,1,8))
    tau[tau < 1e1] = 1e1
    tau[tau > 1e8] = 1e8
    fig,ax = plt.subplots(figsize=(10,6))
    m_iso = pl_acc.planetesimal_iso_mass(Rp)
    m_oli = pl_acc.m_olig(Rp,Mp)
    #cs = ax.contourf(np.log10(X),np.log10(Y),tau,levels=10, locator=ticker.LogLocator(),colors=colors)
    cs = ax.contourf(np.log10(X),np.log10(Y),tau,levels=np.logspace(1,8,8),colors=colors)
    #cs2 = ax.contour(np.log10(X),np.log10(Y),oli_run,levels=[0,1])
    ax.plot(np.log10(Rp),np.log10(m_iso),c='black')
    ax.plot(np.log10(Rp),np.log10(m_oli),linestyle='dashed',c='black')

    def rtom(logr):
        r = 10**logr
        return np.log10(4/3*np.pi* 2*1e5**3 *r**3 / Mearth)

    def mtor(logM):
        M = 10**logM
        return np.log10((np.abs(3/(4*np.pi*2*1e5**3 /Mearth)*M))**(1/3))

    secax = ax.secondary_yaxis('right', functions=(mtor, rtom))

    secax.set_yticks(ticks=[1,2,3,4],labels=[r"$10^1$",r"$10^2$",r"$10^3$",r"$10^4$"])
    secax.set_ylabel('Physical Radius (km)')
    cbar = fig.colorbar(cs,format=ticker.FuncFormatter(fmt),pad=0.11,label='Mass Doubling Timescale (yr)')
    ax.set_xticks(ticks=[0,1,2],labels=[r"$10^0$",r"$10^1$",r"$10^2$"])
    ax.set_yticks(ticks=[-8,-6,-4,-2,0,2],labels=[r"$10^{-8}$",r"$10^{-6}$",r"$10^{-4}$",r"$10^{-2}$",r"$10^0$",r"$10^2$"])
    ax.set_xlabel("Orbital radius (AU)")
    ax.set_ylabel("Mass (Earth Masses)")
    ax.set_title(f"Planetesimal Accretion Ida 2008 at t = {t/yr/1e6} Myr")
    fig.tight_layout()
    
    os.makedirs(f"{location}_drazkowska_plots",exist_ok=True)
    plt.savefig(os.path.join(f"{location}_drazkowska_plots",f"Planeteismal_Accretion_T_{np.round(t/yr/1e6,2)}_Myrs.png"))
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
    timestamp = 0
    iterartor = iter(np.append(np.logspace(0,6,7),2.99e6))
    if restart:
        # Skip evolution already completed
        while not io.finished():
            ti = io.next_event_time()
            
            if ti > model.t:
                break
            else:
                io.pop_events(model.t)
    realtime_i = 0
    delta = 0
    while not io.finished():
        ti = io.next_event_time()
        while (model.t < ti and end==False):
            """
            External photoevaporation - if present, model terminates when all cells at (or below) the base rate as unphysical (and prevents errors).
            Internal photoevaporation - if present, model terminates once the disc is empty.
            Accretion - optionally, the model terminates once unobservably low accretion rates (10^-11 solar mass/year)
            """
            
            if model.num_steps == 0:
                realtime_i = time.time()
            elif model.num_steps == 1000:
                delta = time.time() - realtime_i
                print(f"0 to 1000 steps: {delta}") 
            if model.num_steps < 100 and all_in['planets']['active']:
                save_planets(model, f"{all_in['output']['directory']}_Planet_Data", model.planets, model.t/yr)
            elif model.num_steps < 300 and all_in['planets']['active']:
                if model.num_steps % 10 == 0 and all_in['planets']['active'] == True:
                    save_planets(model, f"{all_in['output']['directory']}_Planet_Data", model.planets, model.t/yr)
            elif all_in['planets']['active']: 
                if model.num_steps % 100 == 0 and all_in['planets']['active'] == True:
                    save_planets(model, f"{all_in['output']['directory']}_Planet_Data", model.planets, model.t/yr)
    
            if False: #model.t/yr >= timestamp and model._planet_model._pl_acc is not None and False:
                drazkowska_plot(model._planet_model._pl_acc,all_in['output']['directory'],model.t)
                timestamp = next(iterartor)
            # External photoevaposration - Read mass loss rates
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
                print(f"Time final: {model.t}")
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
    

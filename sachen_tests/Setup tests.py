
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')) + '/DiscEvolution')

sys.path.append(os.path.abspath(os.path.join('..')) + '/DiscEvolution/planet_formation_scripts')

import importlib

import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import Msun, AU, yr, k_B, m_p,GasConst, Omega0
from DiscEvolution.grid import Grid, MultiResolutionGrid
from DiscEvolution.star import SimpleStar, MesaStar
from DiscEvolution.eos  import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.dust import DustGrowthTwoPop
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolutionFV,ViscousEvolution,LBP_Solution
from DiscEvolution.dust import SingleFluidDrift
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.driver import DiscEvolutionDriver
from DiscEvolution.io import Event_Controller, DiscReader
from DiscEvolution.disc_utils import mkdir_p
from DiscEvolution.planet import Planet, PlanetList
from scipy.special import gamma as gamma_fun

# Try a general setup with Disc Winds
from DiscEvolution.viscous_evolution import TaboneSolution, HybridWindModel

def plotstuff(R,eos,disc,Mdot_visc,Mdot_DW,Mdot_total):
    fig,ax=plt.subplots()
    plt.loglog(R,eos.nu)
    ax.set_ylabel(r'$\nu$ (code units)')
    ax.set_xlabel('Distance (AU)')
    fig,ax=plt.subplots()
    plt.loglog(R,eos.T)
    ax.set_ylabel('T (K)')
    ax.set_xlabel('Distance (AU)')
    fig,ax=plt.subplots()
    plt.loglog(R,disc.Sigma)
    ax.set_ylabel(r'$\Sigma (g/cm^2)$')
    ax.set_xlabel('Distance (AU)')
    fig,ax=plt.subplots()
    plt.loglog(R,eos.cs*AU*yr/3.15e7)
    ax.set_ylabel(r'$c_s (cm/s)$')
    ax.set_xlabel('Distance (AU)')
    fig,ax=plt.subplots()
    plt.loglog(R,Mdot_visc,label='Visc')
    plt.loglog(R,Mdot_DW,label='DW')
    plt.loglog(R,Mdot_total,label='Total')
    plt.legend()
    ax.set_ylabel(r'$\dot{M} (M_\odot/yr)$')
    ax.set_xlabel('Distance (AU)')

Mdisk0 = 0.05
Mdisk = Mdisk0 * Msun # g
Mdot0 = 1.e-8
Mdot = Mdot0 * Msun/yr # g
Rd=35. # AU.  Exponential cutoff of disk.
gamma=1.
mu=2.5
psi=1
# initial guesses
alpha_SS=1.e-3 # initial guess
grid = Grid(0.1, 1000, 1000, spacing='natural')
R = grid.Rc
star = SimpleStar(M=1,R=1,T_eff=4000)
viscous = ViscousEvolutionFV()
hybrid = HybridWindModel(psi)

def sigma_simple(Rd,grid,Mdisk):
    
    Sigma = np.exp(-grid.Rc / Rd) / (grid.Rc / Rd)
    Sigma *= Mdisk / np.trapz(Sigma, np.pi*(grid.Rc*AU)**2)
    return Sigma

# Need initial value of nud to get Sigma solution.  Not used at t=0 though
if False:
    nud=1.
    sol=TaboneSolution(Mdisk/(AU*AU),Rd,nud,psi)
    Sigma = sol(grid.Rc, 0.)
    Om_invsecond=star.Omega_k(R)*yr/3.15e7
elif True:
    Sigma = sigma_simple(Rd,grid,Mdisk)




#eos = IrradiatedEOS(star, alpha_SS)
eos = LocallyIsothermalEOS(star, 0.033, -0.5, alpha_SS)
eos.set_grid(grid)    
eos.update(0,Sigma)
nud = np.interp(Rd, grid.Rc, eos.nu)
eos_cgs=eos.cs*AU*yr/3.15e7

disc = AccretionDisc(grid, star, eos, Sigma)

vr_visc=viscous.viscous_velocity(disc,S=Sigma)
Mdot_visc=np.append(disc.Mdot(vr_visc),0)
#Mdot_DW=3.*np.pi/(Om_invsecond)*Sigma*eos_cgs*eos_cgs*psi*alpha_SS/Msun*3.15e7
Mdot_total=Mdot_visc#+Mdot_DW

#print (Mdot_visc[0],Mdot_DW[0],Mdot_total[0])
#print (disc.Mtot()/Msun)
#plotstuff(R,eos,disc,Mdot_visc,Mdot_DW,Mdot_total)

Case = 1
# Now iterate
# Case 1:  Given Mdisk, Mdot, alpha and psi.  Solve for Sigma (Sigma_0 and rd)
if (Case==1):
    print ('Iterating Case 1')
    for i in range(50):
        # Mdot is proportional to Sigma0.  So scale Sigma
        Sigma=Sigma*Mdot0/Mdot_total[0]
        disc = AccretionDisc(grid, star, eos, Sigma)
        # M is proportional to rc^2.
        Rd=Rd*np.sqrt(Mdisk0/(disc.Mtot()/Msun))
        if False:
            sol=TaboneSolution(Mdisk/(AU*AU),Rd,nud,psi)
            Sigma = sol(grid.Rc, 0.)
        elif True:
            Sigma = sigma_simple(Rd,grid,Mdisk)
        disc = AccretionDisc(grid, star, eos, Sigma)        
        eos.update(0,Sigma)
        #eos_cgs=eos.cs*AU*yr/3.15e7
        vr_visc=viscous.viscous_velocity(disc,S=Sigma)
        Mdot_visc=np.append(disc.Mdot(vr_visc),0)
        #Mdot_DW=3.*np.pi/(Om_invsecond)*Sigma*eos_cgs*eos_cgs*psi*alpha_SS/Msun*3.15e7
        Mdot_total=Mdot_visc#+Mdot_DW
        print (f"Sigma0: {Sigma[0]}, Rd: {Rd}, Mdot0: {Mdot_total[0]}, Mtotal: {disc.Mtot()/Msun}")

    #plotstuff(R,eos,disc,Mdot_visc,Mdot_DW,Mdot_total)    

# Case 2:  Given Mdisk, Mdot, Rd and psi.  Solve for Sigma0 and alpha.
if (Case==2):
    print ('Iterating Case 2')
    for i in range(10):
        # Sigma0 is fully determined by Mdisk and Rd so no need to iterate.
        # Mdot is approximately proportional to alpha.  So scale alpha
        alpha_SS=alpha_SS*(Mdot0/Mdot_total[0])
        eos = IrradiatedEOS(star, alpha_SS, psi*alpha_SS)
        eos.set_grid(grid)    
        eos.update(0,Sigma)
        disc = AccretionDisc(grid, star, eos, Sigma)
        #eos_cgs=eos.cs*AU*yr/3.15e7
        #vr_visc=viscous.viscous_velocity(disc,S=Sigma)
        vr_visc=hybrid.viscous_velocity(disc)
        Mdot_visc=np.append(disc.Mdot(vr_visc),0)
        #Mdot_visc = disc.Mdot(vr_visc)
        #Mdot_DW=3.*np.pi/(Om_invsecond)*Sigma*eos_cgs*eos_cgs*psi*alpha_SS/Msun*3.15e7
        Mdot_total=Mdot_visc#+Mdot_DW
        print (f"Sigma0: {Sigma[0]}, Rd: {disc.RC()}, Mdot0: {Mdot_total[0]}, Mtotal: {disc.Mtot()/Msun}, alpha: {alpha_SS}")
    plotstuff(R,eos,disc,Mdot_visc,Mdot_DW,Mdot_total)  


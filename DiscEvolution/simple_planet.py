import numpy as np
from collections import defaultdict
import json, os

from drift_composition.constants import k_boltzmann, m_hydrogen, G_Msun, Rau, Msun, yr, Mearth
from drift_composition.atoms import molecule_mass, load_protosolar_abundances
from drift_composition.molecule import Molecule

from DiscEvolution.io import DiscReader
from DiscEvolution.chemistry import ChemicalAbund
from DiscEvolution.chemistry.atomic_data import atomic_abundances



def loc_disc (g_val, Rg, dist):
    """
    Interpolates the value of a function at a given distance using linear interpolation.

    Parameters:
    g_val (array-like): Array of function values.
    Rg (array-like): Array of corresponding distances.
    dist (float): The distance at which to interpolate the function value.

    Returns:
    float: Interpolated function value at the given distance.
    """
    cid = np.argmin(np.abs(Rg-dist))
    #loc_val = g_val[cid]
    d_val = (g_val[cid+1]-g_val[cid-1])/(Rg[cid+1]-Rg[cid-1])
    loc_val = g_val[cid] + d_val*(dist-Rg[cid])
    return loc_val

class Planet:
    '''Planet data-object
    
    All masses in Solar masses, distance in cm
    mass, mc, mg: float mass of total, core, gas content
    f_comp: dict key - species, values array of [gas, solid] mass
    '''
    def __init__(self, mass, mc, mg, f_comp, dist=10.0*Rau, time =0.):
        self.mass = mass
        self.mc = mc
        self.mg = mg
        self.dist = dist 
        self.f_comp = f_comp
        self.time = time
    def rhill (self, ms):
        return self.dist*(self.mass/3./ms)**(1./3.)

class PlanetEnv:
    '''
    Local environment around the planet

    All densities sig_ expect and pass g/cm^2, vk = cm/s
    '''
    def __init__(self, grid, alpha, mu, mass_star, gas={'H2': 0.912}, dust={'Si': 3e-5}):
        self.alpha = alpha
        self.mu    = mu #TODO: Check this works the same in disc
        self.mass_star = mass_star
        self.grid  = grid
        self.gas   = gas
        self.dust  = dust

    
    def set_time(self, time):
        pass

    def set_pl_comp(self):
        pass

    def temp(self, T, dist):
        return loc_disc(T, self.grid.Rc, dist) 

    def sig_H2(self, disc, dist):    
        return loc_disc(disc.Sigma_gas, self.grid.Rc, dist) 

    def sig_Si(self, disc, dist):    
        return loc_disc(disc.Sigma_dust, self.grid.Rc, dist) 

    def mols(self, disc): 
        molc = disc.Molecules
        return molc

    def sig_mol(self, disc, dist):
        """
        Calculate the surface density of molecules in the disc at a given distance.
        Parameters:
        -----------
        disc : object
            The disc object containing information about molecules and their surface densities.
        dist : float
            The distance at which to calculate the surface density.
        Returns:
        --------
        sig_mol_g : dict
            Dictionary containing the surface density of gas molecules.
        sig_mol_d : dict
            Dictionary containing the surface density of dust molecules.
        sig_mol_d : dict
            Dictionary containing the surface density of dust molecules (duplicate for compatibility).
        """
        molc = disc.Molecules
        sigma_mol  = disc.Sigma_mol

        sig_mol_d = {}
        sig_mol_g = {}
        for mol in molc:
            s_mol_d = loc_disc(sigma_mol[mol.name][:,1], self.grid.Rc, dist)
            sig_mol_d[mol.name] = s_mol_d

            s_mol_g = loc_disc(sigma_mol[mol.name][:,0], self.grid.Rc, dist)
            sig_mol_g[mol.name] = s_mol_g

        #mol_comp = {
        #    k: np.array([
        #        v[0] + dm_gas*(molg[k]/sg)*dt,
        #        v[1] + (dm_pla+dm_peb)*(mold[k]/sd)*dt
        #    ])
        #    for k, v in planet.f_comp.items()
        #}   

        dust_mean = np.sum([d * molecule_mass(s_d) for s_d, d in self.dust.items()]) / np.sum(list(self.dust.values()))
        gas_mean  = np.sum([g * molecule_mass(s_g) for s_g, g in self.gas.items()]) / np.sum(list(self.gas.values()))
             
        for s_g, a_g in self.gas.items():
            sig_mol_g[s_g]= loc_disc(disc.Sigma_gas, self.grid.Rc, dist)*a_g*molecule_mass(s_g)/gas_mean
            sig_mol_d[s_g]= 0.
        for s_d, a_d in self.dust.items():
            sig_mol_g[s_d]= 0.
            sig_mol_d[s_d]= loc_disc(disc.Sigma_dust, self.grid.Rc, dist)*a_d*molecule_mass(s_d)/dust_mean
        #print(sig_mol_g['H2'], sig_mol_g['He'], self.gas['H2'], gas_mean, molecule_mass('H2'), loc_disc(disc.Sigma_gas, self.grid.Rc, dist))
        return sig_mol_g, sig_mol_d, sig_mol_d

    def sigs_tot(self, disc, dist):
        sg_spec, sd_spec = self.sig_mol(disc, dist)
        sg_tot = np.sum([sg_spec[mol] for mol in list(sg_spec.keys())])
        sd_tot = np.sum([sd_spec[mol] for mol in list(sd_spec.keys())])
        return sg_tot, sd_tot, sd_tot

    def Stokes(self, disc, dist):
        st = disc.Stokes(self.grid.Re)
        if np.isscalar(st):
            stokes = st
        else:
            stokes = loc_disc(st, self.grid.R, dist)
        return stokes

    def vk(self, dist):
        return np.sqrt(self.mass_star*G_Msun/dist)

    def hr(self, T, dist):
        temp = loc_disc(T, self.grid.Rc, dist) 
        cs = np.sqrt(k_boltzmann/self.mu/m_hydrogen*temp)  #TODO: Check mu is used the same way
        return cs/(np.sqrt(self.mass_star*G_Msun/dist))

    @property
    def Rc(self):
        return self.grid.Rc

class PlanetEnv_DiscEvo:
    """
    A class to represent the planetary environment in a disc evolution model.
    Attributes
    ----------
    _reader : DiscReader
        An instance of DiscReader to read the disc data.
    _dtsnap : float
        Time difference between snapshots.
    _mu0 : float, optional
        Initial mean molecular weight.
    _GM : float
        Gravitational constant times the mass of the star.
    mass_star : float
        Mass of the star.
    alpha : float
        Alpha parameter of the disc.
    _grain_abund : defaultdict
        Abundances of grains.
    _ignore_grains : bool
        Flag to ignore grains in calculations.
    _n : int
        Snapshot index.
    _time : float
        Current time.
    _snap : Snapshot
        Current snapshot of the disc.
    _R : ndarray
        Radial positions in the disc.
    _gas_abund : defaultdict
        Abundances of gas species.
    _ice_abund : defaultdict
        Abundances of ice species.
    _pl_abund : dict
        Abundances of planetesimal species.
    mu : float
        Mean molecular weight.
    """
    def __init__(self, DIR, mu=None):
        self._reader = DiscReader(os.path.join(DIR, 'output'))
        self._dtsnap = self._reader[1].time - self._reader[0].time 

        self._mu0 = mu

        self._setup_params(DIR)
        self._setup_abund()

        

    def _setup_params(self,DIR):
        model = json.load(open(os.path.join(DIR, 'DiscConfig.json'), 'r'))
        self._GM = model['star']['mass']*G_Msun
        self.mass_star = model['star']['mass']
        self.alpha = model['disc']['alpha']


    def _setup_abund(self):
        IC = self._reader[0]
        solar = load_protosolar_abundances()

        ignore_grains = 'grain' in IC.chem.ice.names
        
        total = atomic_abundances(IC.chem.gas, ignore_grains=ignore_grains)
        ice = atomic_abundances(IC.chem.ice, ignore_grains=ignore_grains)
        for atom in ice:
            total[atom] += ice[atom]

        nH = total['H'][0]
        if nH < 0.1: nH = 1/1.28
        for atom in total:
            total[atom] /=  nH



        self._grain_abund = defaultdict(float)
        if ignore_grains:
            for atom in total:
                if atom in {'H', 'He', 'Ne', 'Xe', 'Ar'}: 
                    continue

                residual = np.maximum(solar[atom] - total.number_abund(atom), 0)
                self._grain_abund[atom] = residual[0] / IC.chem.ice['grain'][0]
        
            # Add the core refractories if they are missing.
            for elem in ['Si', 'Mg', 'Fe']:
                if elem not in total:
                    self._grain_abund[elem] = solar[elem]/IC.chem.ice['grain'][0]

            # Scale to give mass-abundances
            for elem in self._grain_abund:
                self._grain_abund[elem] *= molecule_mass(elem)

        self._ignore_grains = ignore_grains

        self._n = None 
        self.set_time(0)

    def set_time(self, time):
        self._time = time 

        n = time / self._dtsnap
        if n-int(n) > 0.5:
            n = int(n)+1
        else:
            n = int(n)

        if n == self._n: 
            return

        self._snap = self._reader[n]
        self._R = self._snap.R*Rau

        # Store the abundances of each species.
        ga = atomic_abundances(self._snap.chem.gas, ignore_grains=self._ignore_grains)
        ia = atomic_abundances(self._snap.chem.ice, ignore_grains=self._ignore_grains)

        self._gas_abund = defaultdict(float) | {name:ga[name] for name in ga.names}
        self._ice_abund = defaultdict(float) | {name:ia[name] for name in ia.names}

        if self._ignore_grains:
            for atom in self._grain_abund:
                grains = self._grain_abund[atom]*self._snap.chem.ice['grain']
                self._ice_abund[atom] += grains 

        # Compute the mean molecular weight
        if self._mu0 is None:
            mt, nt = 0, 0
            for mol in self._snap.chem.gas:
                mt += self._snap.chem.gas[mol] 
                nt += self._snap.chem.gas.number_abund(mol)
            self.mu = mt/nt 
        else:
            self.mu = np.full_like(self._R, self._mu0)

    def set_pl_comp(self):
        # Set planetesimal composition to current one
        self._pl_abund = {k:v for k,v in self._ice_abund.items()}

        
    def temp(self, _, dist):
        return loc_disc(self._snap.T, self._R, dist) 

    def sig_H2(self, _, dist):    
        return loc_disc(self._snap.Sigma*(1 - self._snap.dust_frac.sum(0)), self._R, dist) 

    def sig_Si(self, _, dist):    
        return loc_disc(self._snap.Sigma*self._snap.dust_frac.sum(0), self._R, dist)

    def sig_mol(self, disc, dist):

        sig_mol_d = defaultdict(float) 
        sig_mol_g = defaultdict(float) 
        sig_mol_pl = defaultdict(float) 
        for mol in self._gas_abund:
            s_mol_g =  loc_disc(self._snap.Sigma*self._gas_abund[mol], self._R, dist)
            sig_mol_g[mol] = s_mol_g
        
        for mol in self._ice_abund:
            s_mol_d =  loc_disc(self._snap.Sigma*self._ice_abund[mol], self._R, dist)
            sig_mol_d[mol] = s_mol_d
        
        for mol in self._pl_abund:
            s_mol_pl =  loc_disc(self._snap.Sigma*self._pl_abund[mol], self._R, dist)
            sig_mol_pl[mol] = s_mol_pl
        

        return sig_mol_g, sig_mol_d, sig_mol_pl


    def sigs_tot(self, disc, dist):
        sg_spec, sd_spec,  spl_spec = self.sig_mol(disc, dist)
        sg_tot = np.sum([sg_spec[mol] for mol in list(sg_spec.keys())])
        sd_tot = np.sum([sd_spec[mol] for mol in list(sd_spec.keys())])
        spl_tot = np.sum([spl_spec[mol] for mol in list(spl_spec.keys())])
        return sg_tot, sd_tot, spl_tot

    def Stokes(self, disc, dist):
        st = np.pi*self._snap.grain_size[1]/(2*self._snap.Sigma*(1-self._snap.dust_frac.sum(0)))

        return loc_disc(st, self._R, dist)

    def vk(self, dist):
        return np.sqrt(self._GM/dist)

    def hr(self, T, dist):
        cs = loc_disc(np.sqrt(k_boltzmann*self._snap.T/self.mu/m_hydrogen), self._R, dist)
        return cs/self.vk(dist)

    @property
    def Rc(self):
        return self._R

def gap_dens(planet,p_env,disc,T):
    '''Kanagawa 2018, cross flow K description'''
    sig_gas, _, _ = p_env.sigs_tot(disc, planet.dist)
    K = (planet.mass/p_env.mass_star)**2 *pow(p_env.hr(T, planet.dist),-5) / p_env.alpha
    sig_min = sig_gas/(1.0+0.04*K)
    return sig_min

def gas_accretion(planet, p_env, disc, T, f_dm=0.5, f=0.2, kap=0.05, rho_c=5.):
    '''Variation on Bitsch 2015'''
    mass_p = planet.mass
    mc     = planet.mc
    mg     = planet.mg
    dist   = planet.dist
    sig_gas, _, _ = p_env.sigs_tot(disc, dist)
    min_dens    = gap_dens(planet,p_env,disc,T)
    temperature = p_env.temp(T,dist)
    hr          = p_env.hr(T,dist)
    vk          = p_env.vk(dist)

    r_hill = dist*(mass_p/p_env.mass_star/3.)**(1./3.)
    omg_k  = np.sqrt(p_env.mass_star*G_Msun/dist**3)
    if mc > mg:
        dm_gas = (0.00175/f/f/ kap * (rho_c/5.5)**(-1./6.) * np.sqrt(81/temperature)
            *(mc/(Mearth/Msun))**(11./3.) * (0.1*Mearth/Msun / mg) * Mearth/1e6)/Msun
    else:
        dm_disc = -3.0*np.pi*sig_gas*p_env.alpha*hr**2*vk*dist/Msun*yr
        dm_low  = 0.83 * omg_k * min_dens * (hr*dist)**2 * (r_hill/hr/dist)**(4.5) /Msun*yr
        dm_high = 0.14 * omg_k * min_dens * (hr*dist)**2 /Msun*yr
        dm_gas = np.min((dm_low,dm_high,-dm_disc*f_dm))
        #if -dm_disc <np.min((dm_low,dm_high)):
        #print(dist/Rau, -dm_disc*f_dm, np.min((dm_low,dm_high)))
    return dm_gas 

def visc_mig(planet, p_env, disc, T):
    dist   = planet.dist
    temperature = p_env.temp(T,dist)    
    hr          = p_env.hr(T,dist)
    alpha       = p_env.alpha
    r_grid      = p_env.Rc
    ir          = np.argmin(abs(r_grid-dist))
    if ir==1:
        vr = 0.0
    else:
        dr          = abs(r_grid[ir+1] - r_grid[ir-1])
        sig_gas, _, _ = p_env.sigs_tot(disc, r_grid[ir])
        sig_gas_m, _, _ = p_env.sigs_tot(disc, r_grid[ir-1])
        sig_gas_p, _, _ = p_env.sigs_tot(disc, r_grid[ir+1])
        X0          = (dist-dr)**(3./2.)*p_env.hr(T,r_grid[ir-1])**2*p_env.vk(r_grid[ir-1])*sig_gas_m
        X1          = (dist+dr)**(3./2.)*p_env.hr(T,r_grid[ir+1])**2*p_env.vk(r_grid[ir+1])*sig_gas_p
        dr_X        = alpha  * (X1-X0)/(dr)
        if dr_X < 0:
            print('CAUTION:vr>0', dr_X, dist/Rau, sig_gas, ir, planet.mass)
        vr = - abs(3/np.sqrt(dist)/sig_gas *dr_X*yr)
    return vr

def dk_mig(planet, p_env, disc, T):
    '''Migration near visc limit, based on the deviations seen in Durmann 2014
    
    CAUTION: Not suitable for Type I migration object!
    Min speed 0.1 viscous velocity,
    Max speed 10 viscous velocity'''
    dist = planet.dist
    hr   = p_env.hr(T,dist)
    vk   = p_env.vk(dist)
    nu   = p_env.alpha*hr**2*vk*planet.dist
    sig_gas, _, _ = p_env.sigs_tot(disc, dist)
    gas_density = sig_gas/Msun
    sig_std = (1e-7 / yr) /3./np.pi/nu
    v_visc  = visc_mig(planet, p_env, disc, T)

    f_mig   = np.min((4 * (gas_density/sig_std)**(0.6), 5.))
    adot_mig = f_mig* v_visc #* gas_density * dist**2 / 1e-3
    f_still = np.min((0.09*(planet.mass*1e3)**(-0.4), 2.0)) 
    #tau_0   = - gas_density*vk**2*dist**2*(planet.mass/p_env.mass_star/hr)**2 / Msun    
    #ang_mom = planet.mass*dist*vk
    adot0  = f_still * adot_mig
    #factor  = np.max((f_still*f_mig,0.1))
    #a_dot   = factor*v_visc
    #print(a_dot/v_visc, adot0/v_visc, adot_mig/v_visc, f_mig, f_still)
    #if a_dot > 3*v_visc:
        #print(a_dot, v_visc)
    return adot0
    

def plansi_flux (plansi_frac, planet, p_env, disc, T):
    '''Simplified from Fourtier 2013
    
    Notes
    -----
    for planetesimals between (1e-15,1e-7) solar masses collision probability varies between (3.5,1)*1e-3
    Thus sizes dependency is neglected.
    '''
    sig_gas , sig_dust, sig_pl = p_env.sigs_tot(disc, planet.dist)
    massdensity = plansi_frac*sig_pl/Msun
    rh = planet.rhill(p_env.mass_star)
    prob = np.min((3e-2*(Rau/planet.dist)**(1.19),3e-2))#probability (plansi, planet, p_env, disc, T)
    period = 2*np.pi*planet.dist/p_env.vk(planet.dist)/yr
    return 2*np.pi*massdensity*rh**2/period*prob

def pebble_accretion(planet, p_env, disc, T):
    '''Following Bitsch 2015'''
    mass_p = planet.mass
    dist   = planet.dist
    hr        = p_env.hr(T,dist)
    stokes    = np.max((p_env.Stokes(disc,dist),1e-10))
    mass_star = p_env.mass_star
    alpha     = p_env.alpha
    _ , sig_dust, _ = p_env.sigs_tot(disc, dist)
    pebble_density = sig_dust
    r_hill = planet.rhill(mass_star)
    
    v_hill = r_hill * np.sqrt(mass_star*G_Msun / dist**3)
    h_peb = hr * dist / np.sqrt(1 + stokes/alpha)
    dm_2d = min(2.0 * (stokes / 0.1)**(2. / 3.) * r_hill * v_hill * pebble_density,2.0 * r_hill * v_hill * pebble_density)
    dm_3d = dm_2d * (r_hill * np.pi**0.5 / 2**1.5 / h_peb *(stokes/0.1)**(1./3.))
    crit_h = np.pi* (stokes/0.1)**(1./3.) * r_hill /2/np.sqrt(2*np.pi)
    if h_peb > crit_h: dm_peb = dm_2d
    else: dm_peb = dm_3d
    if planet.mass > 20 * (p_env.hr(T, dist)/0.05)**3. * Mearth/Msun:
        dm_peb = 0
    return dm_peb/Msun*yr

def mass_growth_pl(planet, p_env, disc, T, dt, plansi_frac):
    dist = planet.dist

    dm_pla = plansi_flux(plansi_frac, planet, p_env, disc, T)
    dm_peb = pebble_accretion(planet, p_env, disc, T)
    dm_gas = gas_accretion(planet, p_env, disc, T)
    mc = planet.mc + dm_pla*dt + dm_peb*dt
    mg = planet.mg + dm_gas*dt
    
    sg , sd, spl = p_env.sigs_tot(disc, dist)
    molg, mold, molpl = p_env.sig_mol(disc,dist)
    mol_names = list(molg.keys())
    
    mol_comp = {
        k: np.array([
            v[0] + dm_gas*(molg[k]/sg)*dt,
            v[1] + (dm_pla*molpl[k]/spl+dm_peb*mold[k]/sd)*dt
        ])
        for k, v in planet.f_comp.items()
    }

    new_planet = Planet(mc+mg, mc, mg, mol_comp, planet.dist, planet.time+dt)
    return new_planet

def mig_planet(planet, p_env, disc, T, dt):
    a_dot = dk_mig(planet,p_env, disc,T)
    #a_dot = visc_mig(planet, p_env, disc, T)
    return planet.dist + a_dot*dt

class FixedMigration:
    def __init__(self, t_mig, M0=Mearth, indx=0):
        self._t_mig = t_mig
        self._M0 = M0
        self._k = indx
    
    def __call__(self, planet, p_env, disc, T, dt):
        t_mig = self._t_mig*(planet.mass/self._M0)**self._k
        return planet.dist * np.exp(-dt/t_mig)

    def adot(self, planet, p_env, disc, T):
        t_mig = self._t_mig*(planet.mass/self._M0)**self._k
        return - planet.dist / t_mig

def std_evo_comp(planet_in, DM, p_env, T, f_plansi, dt_ini, nt, final_radius = 0.1, final_time=3e6, migration=None):
    planet_evo = np.array([planet_in])
    r_grid     = p_env.Rc
    dt_adapt   = dt_ini
    t          = planet_in.time
    nn = 0

    if migration is None:
        migration = mig_planet
        adot = dk_mig
    else:
        adot = migration.adot

    p_env.set_time(t)
    p_env.set_pl_comp()

    if final_radius*Rau < r_grid[0]:
        final_radius = 1.1* r_grid[0]/Rau
        print('reset final radius to: ', final_radius)
    for nn in range(nt-1):
        t += dt_adapt
        p_env.set_time(t)

        planet = mass_growth_pl(planet_in, p_env, DM, T, dt_adapt, f_plansi) 
        planet.dist = np.max((migration(planet, p_env, DM, T, dt_adapt) ,1e-4*Rau))
        ir = np.argmin(abs(r_grid-planet.dist))
        dr = r_grid[ir+1]-r_grid[ir]
        dt_adapt = min((abs(dr / adot(planet, p_env, DM, T))*0.1,
                        abs(planet_in.mass*5e-2 / gas_accretion(planet_in, p_env, DM, T))*0.1,
                        10000))
        dt_adapt = max((dt_adapt, 100))
        #print('dt=',dt_adapt, 
        #        '; dx=', dr/Rau,
        #        '; r=', planet.dist/Rau, 
        #        '; dm=', gas_accretion(planet, p_env, DM,  T), 
        #        abs(dr / dk_mig(planet, p_env, DM, T))*0.5,
        #        abs(planet_in.mass*5e-2 / gas_accretion(planet_in, p_env, DM, T))*0.5)
        if nn%10==0:
            planet_evo = np.append(planet_evo, planet)
        planet_in = planet
        if planet.dist < final_radius*Rau:
            planet.dist = final_radius*Rau # Halt migration at inner edge
        if planet.mass > 0.3e-3:
            print('0.3Mjup at t = {}Myr; n= {}'.format(t/1e6,nn))
            break
        elif t > final_time:
            print('{} Myr evolution reached ; n= {}'.format(t/1e6,nn))
            break
    #print(nn,len(planet_evo))
    return planet_evo, len(planet_evo)

def no_mig_evo_comp(planet_in, DM, p_env, T, f_plansi, dt_ini, nt, final_time=3e6):
    planet_evo = np.array([planet_in])
    r_grid     = p_env.grid.Rc
    dt_adapt   = dt_ini
    t          = 0.
    nn = 0
    for nn in range(nt-1):
        t += dt_adapt
        p_env.set_time(t)

        planet = mass_growth_pl(planet_in, p_env, DM, T, dt_adapt, f_plansi) 
        ir = np.argmin(abs(r_grid-planet.dist))
        dr = r_grid[ir+1]-r_grid[ir]
        dt_adapt = min((abs(planet_in.mass*5e-2 / gas_accretion(planet_in, p_env, DM, T))*0.5,
                        10000))
        dt_adapt = max((dt_adapt, 100))
        if nn%10==0:
            planet_evo = np.append(planet_evo, planet)
        planet_in = planet
        if planet.mass > 0.3e-3:
            print('0.3Mjup at t = {}; n= {}, r={}'.format(t,nn,planet.dist/Rau))
            break
        elif t > final_time:
            print('{} yr evolution reached ; n= {}, r={}'.format(t,nn,planet.dist/Rau))
            break
    #print(nn,len(planet_evo))
    return planet_evo[:nn//10], nn//10

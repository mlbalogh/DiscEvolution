"""
run_model_student.py
=====================

A compact, heavily-commented walkthrough of a single DiscEvolution run,
written for someone seeing this codebase for the first time.

WHAT THIS RUNS
--------------
This script reproduces exactly one physical setup out of the several that
DiscEvolution supports: a viscously- and magnetically-(disc-wind)-accreting
disc, with two-population dust growth and radial drift, simple C/O
chemistry, planetesimal formation, and Bitsch-model planet growth/migration.

UNIT CONVENTIONS (the part that trips everyone up at first)
-------------------------------------------------------------
DiscEvolution works in units where G = 1, with length in AU and mass in
Msun (see DiscEvolution/constants.py). Those three choices fix the time
unit too, via Kepler's third law -- and it is *not* seconds or years, it's
whatever comes out of G = AU = Msun = 1. The constant `yr` (=2*pi) is the
conversion factor: multiply a duration in real years by `yr` to get the
equivalent duration `t` in code-time units, and divide a code-time `t` by
`yr` to get real years back. That's why you'll see a lot of `* yr` and
`/ yr` scattered through this file, and why the disc-orbital angular
frequency Omega_k comes out looking like it's "in units of 2*pi/yr".

Other quantities you'll meet below and their units:
    R, Rd, grid.Rc      radius                      AU
    Sigma               surface density             g / cm^2
    M (disc/planet)     mass                         Msun, unless stated
                                                       otherwise (planet
                                                       core/env masses are
                                                       tracked in Mearth --
                                                       see planet_formation.py)
    Mdot                accretion rate               Msun / yr
    T                   temperature                  K
    alpha, alpha_SS      viscosity parameters         dimensionless
    t (this script)      simulation time              code-time units
                                                       (divide by `yr` for
                                                       years)

PIPELINE OVERVIEW
-----------------
    1. Load the JSON config (and any --flag overrides from the CLI).
    2. Build the grid and star.
    3. Solve for the initial disc structure (disc_setup.setup_disc).
    4. Attach gas/dust transport (viscous+wind evolution, radial drift,
       turbulent diffusion) and wrap the disc in a DustGrowthTwoPop object.
    5. Seed the disc chemistry in equilibrium with the dust.
    6. Turn on planetesimal formation (optional).
    7. Place planets and attach the planet-growth model (optional).
    8. Open an HDF5 file and stream results to it as the simulation runs.
    9. Integrate forward in time, writing a snapshot at each requested
       output time.

Run it with:
    python run_model_student.py --config config/DiscConfig_default.json \\
        --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50
"""

import os
import sys
import json
import time

import numpy as np
import h5py

from DiscEvolution.constants import AU, Msun, yr
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolutionFV, HybridWindModel
from DiscEvolution.dust import DustGrowthTwoPop, SingleFluidDrift, PlanetesimalFormation
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.planet_formation import Planets, Bitsch2015Model
from DiscEvolution.chemistry import (
    SimpleCOChemOberg, EquilibriumCOChemOberg, TimeDepCOChemOberg, SimpleCOAtomAbund,
)

from disc_setup import setup_disc

GAS_SOLVER = ViscousEvolutionFV   # viscous-evolution scheme used when winds are off


# ============================================================================
# Ice lines (used only for diagnostics written to the output file)
# ============================================================================

def compute_ice_lines(chem, grid, threshold=0.5):
    """
    Compute ice line radius for each molecular species.

    Parameters
    ----------
    chem : MolecularIceAbund
        Chemistry object with gas and ice abundances for each species
    grid : Grid
        Grid object with Rc radial positions (in AU)
    threshold : float
        Condensation fraction threshold (0-1) to define ice line. Default=0.5

    Returns
    -------
    ice_lines : ndarray of shape (Nspec,)
        Ice line radius (AU) for each species. NaN if no clear transition.
    """
    Nspec = chem.ice.Nspec
    ice_lines = np.full(Nspec, np.nan)

    for i, species in enumerate(chem.ice.names):
        ice_abund = chem.ice.data[i, :]
        gas_abund = chem.gas.data[i, :]
        total_abund = ice_abund + gas_abund

        if np.max(total_abund) < 1e-300:
            continue

        ice_fraction = np.divide(ice_abund, total_abund,
                                  where=total_abund > 0,
                                  out=np.zeros_like(ice_abund))

        min_frac = np.min(ice_fraction[total_abund > 0]) if np.any(total_abund > 0) else 0
        max_frac = np.max(ice_fraction[total_abund > 0]) if np.any(total_abund > 0) else 0
        if (max_frac - min_frac) < 0.1:
            continue

        crossing = np.diff(np.sign(ice_fraction - threshold))
        cross_idx = np.where(crossing != 0)[0]
        if len(cross_idx) > 0:
            idx = cross_idx[0]
            f1, f2 = ice_fraction[idx], ice_fraction[idx + 1]
            r1, r2 = grid.Rc[idx], grid.Rc[idx + 1]
            if abs(f2 - f1) > 1e-10:
                ice_lines[i] = r1 + (threshold - f1) / (f2 - f1) * (r2 - r1)
            else:
                ice_lines[i] = r1

    return ice_lines


# ============================================================================
# Step 2: time grid
# ============================================================================

def make_time_grid(sim_params):
    """
    Build the array of simulation times (in code-time units) at which the
    disc state will be checkpointed to the output file.

    sim_params['t_interval'] can be given three ways:
        "power"   -- log-spaced snapshots from t_initial to t_final (years)
        a list    -- explicit snapshot times, in Myr
        a number  -- fixed linear spacing, in years, from t_initial to t_final
    """
    t_interval = sim_params['t_interval']

    if t_interval == "power":
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            years = np.logspace(0, np.log10(sim_params['t_final']), num=num_points)
        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            years = np.logspace(np.log10(sim_params['t_initial']),
                                 np.log10(sim_params['t_final']), num=num_points)
        return years * yr

    elif isinstance(t_interval, list):
        Myr = np.array(t_interval)
        return Myr * 1e6 * yr

    else:
        years = np.arange(sim_params['t_initial'], sim_params['t_final'], t_interval)
        return years * yr


# ============================================================================
# Step 4: gas/dust transport + dust-growth disc wrapper
# ============================================================================

def build_transport(transport_params, wind_params, disc_params, dust_growth_params, lambda_DW):
    """
    Build the operators that move gas and dust around the disc each
    timestep. Any of the three can be turned off independently via
    `transport_params` (e.g. to hold the gas disc fixed while testing
    chemistry).
    """
    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW)
        else:
            gas = GAS_SOLVER()

    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        # SingleFluidDrift does its own internal diffusion call when given
        # a `diffusion` object, so hand ours off and stop calling it
        # separately further down.
        dust = SingleFluidDrift(diffusion=diffuse,
                                 settling=dust_growth_params['settling'],
                                 van_leer=transport_params['van_leer'])
        diffuse = None

    return gas, dust, diffuse


def build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas):
    """Wrap the bare (grid, star, eos, Sigma) disc in dust-growth physics."""
    return DustGrowthTwoPop(
        grid, star, eos, disc_params['d2g'], Sigma=Sigma,
        feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
        f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
        uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas,
    )


# ============================================================================
# Step 5: chemistry
# ============================================================================

_CHEM_MODELS = {
    "Simple": lambda: SimpleCOChemOberg(),
    "Equilibrium": lambda: EquilibriumCOChemOberg(a=1e-5),
    "Equilibrium_Fixed": lambda: EquilibriumCOChemOberg(a=1e-5, fix_ratios=True),
    "TimeDep": lambda: TimeDepCOChemOberg(a=1e-5),
}


def build_chemistry(disc, chemistry_params, d2g_target, N_cell):
    """
    Seed the disc with ice/gas-phase chemical abundances in equilibrium
    with the initial dust-to-gas ratio, iterating a few times because the
    ice fraction and the dust-to-gas ratio depend on each other.

    Returns (chemistry_model, Natom, Nmol). Also sets disc.chem and
    initializes disc.dust_frac from the ice abundances.
    """
    if not chemistry_params["on"]:
        disc.chem = None
        return None, 1, 1  # dummy dimensions when chemistry is off

    try:
        chemistry = _CHEM_MODELS[chemistry_params["chem_model"]]()
    except KeyError:
        raise ValueError("Valid chemistry model not selected. "
                          "Choose Simple, Equilibrium, Equilibrium_Fixed, or TimeDep")

    # Solar reference abundances (number of atoms per H), same for every cell.
    X_solar = SimpleCOAtomAbund(N_cell)
    X_solar.set_solar_abundances()

    # The dust-to-gas ratio depends on how much ice has condensed, and the
    # equilibrium ice fraction depends on the dust-to-gas ratio (it sets the
    # dust surface area available for condensation) -- so iterate.
    chem = None
    for _ in range(100):
        if chemistry_params["assert_d2g"]:
            # Rescale the dust fraction so the *total* dust-to-gas ratio
            # matches disc_params['d2g'] exactly, rather than whatever the
            # ice chemistry alone would produce.
            M_dust = np.trapz(disc.Sigma_D.sum(0), np.pi * disc.grid.Rc ** 2)
            M_gas = np.trapz(disc.Sigma_G, np.pi * disc.grid.Rc ** 2)
            mod_frac = d2g_target / (M_dust / M_gas)
            disc.dust_frac[:] = disc.dust_frac * mod_frac

        dust_frac = disc.dust_frac.sum(0)
        chem = chemistry.equilibrium_chem(disc.T, disc.midplane_gas_density, dust_frac, X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)

    disc.chem = chem
    disc.update_ices(disc.chem.ice)

    Natom = disc.chem.ice.atomic_abundance().data.shape[0]
    Nmol = disc.chem.gas.data.shape[0]
    return chemistry, Natom, Nmol


# ============================================================================
# Step 6: planetesimals
# ============================================================================

def build_planetesimals(disc, planetesimal_params):
    """Attach a PlanetesimalFormation object to the disc, if enabled."""
    disc._planetesimal = None
    if planetesimal_params['active']:
        disc._planetesimal = PlanetesimalFormation(
            disc,
            d_planetesimal=planetesimal_params['diameter'],
            St_min=planetesimal_params['St_min'],
            St_max=planetesimal_params['St_max'],
            pla_eff=planetesimal_params['pla_eff'],
        )


# ============================================================================
# Step 7: planets
# ============================================================================

def build_planets(disc, planet_params, chemistry_params, wind_params):
    """
    Create the Planets container and the Bitsch2015Model that grows/
    migrates them, and drop each planet in at its starting radius and mass.

    Planets start with a bare core and no envelope (X_env = 0); their core
    composition is read off the disc's ice abundance at the planet's
    starting radius.
    """
    if not planet_params['include_planets']:
        return None, None

    if chemistry_params["on"]:
        Nchem = disc.chem.ice.data.shape[0]
        planets = Planets(Nchem=Nchem)
    else:
        planets = Planets(Nchem=0)

    planet_model = Bitsch2015Model(
        disc, pb_gas_f=planet_params["pb_gas_f"],
        migrate=planet_params["migrate"],
        pebble_acc=planet_params["pebble_accretion"],
        gas_acc=planet_params["gas_accretion"],
        planetesimal_acc=planet_params["planetesimal_accretion"],
        winds=wind_params["on"],
    )
    planet_model.set_disc(disc)

    for Rp_i, Mp_i, t_implant in zip(planet_params['Rp'], planet_params['Mp'],
                                      planet_params['implant_time']):
        if chemistry_params["on"]:
            X_core = np.array([
                disc.interp(Rp_i, ice_spec) / disc.interp(Rp_i, disc.dust_frac[:2].sum(0))
                for ice_spec in disc.chem.ice.data
            ])
            X_env = np.zeros_like(X_core)
            planets.add_planet(t_implant, Rp_i, Mp_i, 0, X_core, X_env)
        else:
            planets.add_planet(t_implant, Rp_i, Mp_i, 0)

    return planets, planet_model


# ============================================================================
# HDF5 streaming output
# ============================================================================
#
# Every quantity below is stored as a "growable" dataset: created with an
# initial length of 0 along axis 0, then extended by one row per snapshot
# with `grow_and_set`. This keeps the file readable by
# run_model_stream.load_visc_data() (and the analysis notebooks that use
# it) whether the run is 10 steps or 10 million steps long, and without
# needing to know the final length in advance.
#
# Keep the dataset/group *names* exactly as they are here -- the loader
# and existing notebooks key off them by name.

def grow_and_set(dset, value):
    """Append one row to a growable HDF5 dataset."""
    n = dset.shape[0]
    dset.resize(n + 1, axis=0)
    dset[n] = value


def create_output_file(outfile, grid, config, Natom, Nmol, alpha_SS):
    """
    Create the HDF5 file and every dataset/group it will need, but do not
    write any data yet (see write_planet_row / write_disc_snapshot below).
    Returns the open h5py.File plus a dict of the per-planet group handles.
    """
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    nR = len(grid.Rc)

    h5f = h5py.File(outfile, "w")
    h5f.attrs["alpha_SS"] = float(alpha_SS)

    # ---- scalar time series (one number per snapshot) ----
    for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
        h5f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

    # ---- per-planet time series ----
    groups = {}
    if planet_params['include_planets']:
        grp_Mcs = h5f.create_group("Mcs")
        grp_Mes = h5f.create_group("Mes")
        grp_Rp = h5f.create_group("Rp")
        grp_Mdotp = h5f.create_group("disk_Mdot_p")
        grp_Xc = h5f.create_group("X_cores")
        grp_Xe = h5f.create_group("X_envs")
        grp_ice_lines = h5f.create_group("ice_lines")
        grp_M_transition = h5f.create_group("M_transition")
        grp_M_iso = h5f.create_group("M_iso")

        nplanets = len(planet_params["Mp"])
        for ip in range(nplanets):
            grp_Mcs.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Mes.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Rp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Mdotp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_ice_lines.create_dataset(str(ip), shape=(0, Nmol), maxshape=(None, Nmol),
                                          dtype="f8", chunks=(1024, Nmol))
            grp_M_transition.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_M_iso.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))

            pgrp_c = grp_Xc.create_group(str(ip))
            pgrp_e = grp_Xe.create_group(str(ip))
            if chemistry_params["on"]:
                nchem = Nmol  # one dataset per ice species tracked on the planet
                for js in range(nchem):
                    pgrp_c.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    pgrp_e.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))

        groups = dict(Mcs=grp_Mcs, Mes=grp_Mes, Rp=grp_Rp, disk_Mdot_p=grp_Mdotp,
                      X_cores=grp_Xc, X_envs=grp_Xe, ice_lines=grp_ice_lines,
                      M_transition=grp_M_transition, M_iso=grp_M_iso)

    # ---- grid (written once) ----
    h5f.create_dataset("R", data=grid.Rc)

    # ---- disc profile snapshots (one row of length nR per snapshot time) ----
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
    h5f.create_dataset("disk_ice_lines", shape=(0, Nmol), maxshape=(None, Nmol), dtype="f8")
    h5f.create_dataset("T", shape=(0, nR), maxshape=(None, nR), dtype="f8")

    if planetesimal_params["active"]:
        h5f.create_dataset("Sigma_planetesimals", shape=(0, nR), maxshape=(None, nR), dtype="f8")
        h5f.create_dataset("disk_planetesimal_atom_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
        h5f.create_dataset("disk_planetesimal_mol_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")

    return h5f, groups


def write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                      chemistry_params, dust_growth_params):
    """Append one row to every per-planet dataset (called every 5 steps)."""
    for ip, planet in enumerate(planets):
        grow_and_set(groups["Mcs"][str(ip)], planet.M_core.copy())
        grow_and_set(groups["Mes"][str(ip)], planet.M_env.copy())
        grow_and_set(groups["Rp"][str(ip)], planet.R.copy())
        grow_and_set(groups["disk_Mdot_p"][str(ip)], np.interp(planet.R, grid.Rc[0:-1], disk_Mdot))

        if planet_model._peb_acc:
            grow_and_set(groups["M_transition"][str(ip)], planet_model._peb_acc.M_transition(planet.R))
            grow_and_set(groups["M_iso"][str(ip)], planet_model._peb_acc.M_iso(planet.R))

        if chemistry_params["on"]:
            for js, chem in enumerate(planet.X_core):
                grow_and_set(groups["X_cores"][str(ip)][str(js)], chem)
            for js, env in enumerate(planet.X_env):
                grow_and_set(groups["X_envs"][str(ip)][str(js)], env)

            ice_lines = compute_ice_lines(disc.chem, disc.grid, threshold=dust_growth_params['thresh'])
            grow_and_set(groups["ice_lines"][str(ip)], ice_lines)


def write_disc_snapshot(h5f, disc, config, t, Natom, Nmol):
    """Append one row to every disc-profile dataset (called once per output time)."""
    chemistry_params = config["chemistry"]
    planetesimal_params = config["planetesimal"]

    grow_and_set(h5f["time_snap"], t / (1e6 * yr))  # Myr
    grow_and_set(h5f["Sigma_G"], disc.Sigma_G)
    grow_and_set(h5f["Sigma_dust"], disc.Sigma_D[0])
    grow_and_set(h5f["Sigma_pebbles"], disc.Sigma_D[1])
    grow_and_set(h5f["Sigma_pebble_size"], disc.grain_size[1])
    grow_and_set(h5f["T"], disc.T)
    grow_and_set(h5f["Vdrift"], disc.v_drift)

    if chemistry_params["on"]:
        grow_and_set(h5f["disk_atom_gas_abund"], disc.chem.gas.atomic_abundance().data)
        grow_and_set(h5f["disk_mol_gas_abund"], disc.chem.gas.data)
        grow_and_set(h5f["disk_atom_ice_abund"], disc.chem.ice.atomic_abundance().data)
        grow_and_set(h5f["disk_mol_ice_abund"], disc.chem.ice.data)
        grow_and_set(h5f["disk_ice_lines"],
                     compute_ice_lines(disc.chem, disc.grid, threshold=config["dust_growth"]["thresh"]))
    else:
        nR = len(disc.grid.Rc)
        grow_and_set(h5f["disk_atom_gas_abund"], np.zeros((Natom, nR)))
        grow_and_set(h5f["disk_mol_gas_abund"], np.zeros((Nmol, nR)))
        grow_and_set(h5f["disk_atom_ice_abund"], np.zeros((Natom, nR)))
        grow_and_set(h5f["disk_mol_ice_abund"], np.zeros((Nmol, nR)))

    if planetesimal_params["active"]:
        grow_and_set(h5f["Sigma_planetesimals"], disc.Sigma_D[2])
        if chemistry_params["on"]:
            grow_and_set(h5f["disk_planetesimal_atom_abund"], disc._planetesimal.ice_abund.atomic_abundance().data)
            grow_and_set(h5f["disk_planetesimal_mol_abund"], disc._planetesimal.ice_abund.data)


# ============================================================================
# Output filename
# ============================================================================
#
# This is the ONE place that decides the output filename, so it can never
# drift out of sync with a copy hardcoded somewhere else (that used to
# happen between this script and run_popsynth_parallel.sh). The prefix
# comes from config['simulation']['run_name'] -- change it there and every
# script that calls output_filename() picks it up automatically. A batch
# launcher does not need to know this format at all: it can just always
# invoke run_model_student.py and let `run_model()`'s own skip-if-complete
# check (below) decide whether there's anything to do.

def output_filename(config):
    """Build the deterministic output filename for one parameter combination."""
    sim_params = config['simulation']
    disc_params = config['disc']
    wind_params = config['winds']
    run_name = sim_params.get('run_name', 'run')
    return (f"{run_name}_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}"
            f"_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5")


# ============================================================================
# Main driver
# ============================================================================

def run_model(config, cli_output_dir=None):
    """
    Run one disc-evolution simulation from start to finish and stream the
    result to an HDF5 file.

    Parameters
    ----------
    config : dict
        Parsed JSON configuration (see config/DiscConfig_default.json for
        an example of every field used below).
    cli_output_dir : str, optional
        Output directory, highest priority (beats $DISCEVOLUTION_OUTPUT
        and config['simulation']['output_dir']).
    """
    grid_params = config['grid']
    sim_params = config['simulation']
    star_params = config['star']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    wind_params = config['winds']

    # ---- 0. skip immediately if this exact run already finished ----
    # (checked before any of the expensive setup below, so a batch sweep
    #  can be safely re-launched after an interruption without re-solving
    #  discs it already has answers for)
    output_dir = cli_output_dir or os.environ.get(
        'DISCEVOLUTION_OUTPUT', sim_params.get('output_dir', './output'))
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, output_filename(config))
    if os.path.exists(outfile):
        with h5py.File(outfile, "r") as existing:
            if existing.attrs.get("complete", False):
                print(f"Skipping -- output already complete: {outfile}")
                return
        print(f"Output file exists but is incomplete; re-running: {outfile}")

    # ---- 2. grid + star ----
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'],
                spacing=grid_params['spacing'])
    star = SimpleStar(M=star_params["M"], R=star_params["R"], T_eff=star_params['T_eff'])
    times = make_time_grid(sim_params)

    opacity_tables = {"Tazzari": Tazzari2016, "Zhu2012": Zhu2012}
    kappa = opacity_tables.get(eos_params["opacity"], Zhu2012)

    # ---- 3. solve for the initial disc structure ----
    disc, eos, Sigma, alpha, alpha_SS, lambda_DW = setup_disc(grid, star, config, kappa)

    # Sanity check: outside this range the alpha-solve above is not
    # meaningful (either essentially inviscid, or so viscous the disc
    # wouldn't survive), so there's no point integrating it forward.
    if (alpha_SS > 0.1) or (alpha_SS < 1e-5):
        print(f"Not running model - alpha_SS out of range. "
              f"alpha={eos.alpha}, Rd={disc_params['Rd']}, Mdisk={disc.Mtot()/Msun:.4g} Msun")
        return
    print(f"Running model. alpha={eos.alpha}, Rd={disc_params['Rd']}, "
          f"Mdisk={disc.Mtot()/Msun:.4g} Msun")

    # ---- 4. transport + dust growth ----
    gas, dust, diffuse = build_transport(transport_params, wind_params, disc_params,
                                          dust_growth_params, lambda_DW)
    disc = build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas)

    # ---- 5. chemistry ----
    chemistry, Natom, Nmol = build_chemistry(disc, chemistry_params, disc_params["d2g"],
                                              grid_params["nr"])

    # ---- 6. planetesimals ----
    build_planetesimals(disc, planetesimal_params)

    # ---- 7. planets ----
    planets, planet_model = build_planets(disc, planet_params, chemistry_params, wind_params)
    nplanets = len(planet_params["Mp"])

    # ---- 8. output file (path already computed in the skip-check above) ----
    h5f, groups = create_output_file(outfile, grid, config, Natom, Nmol, alpha_SS)
    try:
        _integrate(h5f, groups, disc, grid, star, planets, planet_model, gas, dust, diffuse,
                   chemistry, times, config)
    finally:
        h5f.attrs["complete"] = True
        h5f.close()


def _disc_star_mdot(disc):
    """Accretion rate onto the star at the current disc state, in Msun/yr."""
    v = disc._gas.viscous_velocity(disc, disc.Sigma)
    Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * v * (AU * AU) * (yr / Msun)
    return Mdot


def _integrate(h5f, groups, disc, grid, star, planets, planet_model, gas, dust, diffuse,
               chemistry, times, config):
    """The time-stepping loop, plus the periodic writes to `h5f`."""
    transport_params = config['transport']
    chemistry_params = config['chemistry']
    planet_params = config['planets']
    dust_growth_params = config['dust_growth']
    planetesimal_params = config['planetesimal']
    Natom, Nmol = h5f["disk_atom_gas_abund"].shape[1], h5f["disk_mol_gas_abund"].shape[1]

    # ---- t = 0 row, if the requested snapshot times don't already include it ----
    if 0.0 not in config['simulation']['t_interval']:
        disk_Mdot = _disc_star_mdot(disc)
        grow_and_set(h5f["t"], 0.0)
        grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
        grow_and_set(h5f["disk_Mass"], disc.Mtot())
        grow_and_set(h5f["Tc"], disc.T[0])
        grow_and_set(h5f["Sigc"], disc.Sigma[0])
        if planets is not None:
            write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                              chemistry_params, dust_growth_params)
        write_disc_snapshot(h5f, disc, config, 0.0, Natom, Nmol)
        h5f.flush()

    # ---- bookkeeping for progress / ETA reporting ----
    t, n = 0.0, 0
    wall_spent = 0.0
    eta_abort_hours = config['simulation'].get('eta_abort_hours', 24)
    last_eta_hours, last_eta_minutes = 0, 0

    for ti in times:
        while t < ti:
            step_start = time.time()

            # Physics-limited timestep (each active operator proposes the
            # largest dt it can safely take; we use the smallest, further
            # capped so we land exactly on the next requested snapshot).
            dt_physics = float('inf')
            if transport_params['gas_transport']:
                dt_physics = min(dt_physics, disc._gas.max_timestep(disc))
            if transport_params['radial_drift']:
                dt_physics = min(dt_physics, dust.max_timestep(disc))
            dt = min(ti - t, dt_physics)

            # Rough ETA, recomputed every 1000 steps, used to bail out of
            # runs that would take unreasonably long (useful when sweeping
            # a large parameter grid in parallel -- see run_popsynth_parallel.sh).
            if n >= 1000 and (n % 1000) == 0:
                avg_sec_per_step = wall_spent / n
                dt_for_eta = dt_physics if dt_physics < float('inf') else dt
                remaining_steps = int(np.ceil(max(times[-1] - t, 0.0) / max(dt_for_eta, 1e-300)))
                eta_seconds = avg_sec_per_step * remaining_steps
                last_eta_hours = int(eta_seconds // 3600)
                last_eta_minutes = int((eta_seconds % 3600) // 60)
                if eta_seconds > eta_abort_hours * 3600:
                    print(f"\n=== WARNING: ETA {last_eta_hours:02d}:{last_eta_minutes:02d} (h:m) "
                          f"exceeds threshold ({eta_abort_hours}h); stopping early at "
                          f"t = {t/(yr*1e6):.6f} Myr (target was {times[-1]/(yr*1e6):.6f} Myr) ===",
                          file=sys.stderr)
                    return

            dust_frac = getattr(disc, "dust_frac", None)
            gas_chem = disc.chem.gas.data if chemistry_params["on"] else None
            ice_chem = disc.chem.ice.data if chemistry_params["on"] else None

            # --- gas viscous/wind evolution and dust radial drift ---
            # (tracer arrays are passed in/out so gas/dust transport also
            #  advects the chemical abundances along with the mass)
            if transport_params['gas_transport']:
                dust_frac_for_gas = dust_frac[:-1] if disc._planetesimal else dust_frac
                disc._gas(dt, disc, [dust_frac_for_gas, gas_chem, ice_chem])
            if transport_params['radial_drift']:
                dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            # --- turbulent diffusion (only if not already folded into `dust`) ---
            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)
                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem)
                if dust_frac is not None:
                    dust_slice = dust_frac[:2] if disc._planetesimal else dust_frac[:]
                    dust_slice += dt * diffuse(disc, dust_slice)

            # --- enforce physical bounds (transport can produce small
            #     negative overshoots near sharp gradients) ---
            disc.Sigma[:] = np.maximum(disc.Sigma, 0)
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
            if chemistry_params["on"]:
                disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

            # --- planetesimal formation (converts drifting pebbles to a
            #     dynamically-decoupled planetesimal population) ---
            if disc._planetesimal:
                disc._planetesimal.update(dt, disc, dust)

            # --- chemistry: adsorption/desorption given the new T, rho,
            #     dust surface area ---
            if chemistry_params["on"]:
                dust_frac_for_chem = disc.dust_frac[:-1].sum(0) if disc._planetesimal else disc.dust_frac.sum(0)
                chemistry.update(dt, disc.T, disc.midplane_gas_density, dust_frac_for_chem, disc.chem)
                disc.update_ices(disc.chem.ice)

            # --- planet growth/migration (pebble + gas accretion, disc torques) ---
            if planets is not None:
                planet_model.integrate(dt, planets)

            # --- advance the disc's own bookkeeping (grain sizes, etc.) ---
            disc.update(dt)

            t += dt
            n += 1
            wall_spent += time.time() - step_start

            if (n % 1000) == 0:
                print(f"\rNstep: {n}", flush=True)
                print(f"\rTime: {t/(1e6*yr)} Myr", flush=True)
                print(f"\rdt: {dt/yr} yr", flush=True)
                print(f"\rETA: {last_eta_hours:02d}:{last_eta_minutes:02d} (h:m)", flush=True)

            # --- stream disc-level (and, if present, per-planet) series every 5 steps ---
            # These scalar series are useful on their own even with no planets
            # (e.g. Md(t), Mdot(t) for a disc-only run), so they're written
            # unconditionally; only the per-planet groups need `planets`.
            if (n % 5 == 0):
                disk_Mdot = _disc_star_mdot(disc)
                grow_and_set(h5f["t"], t / yr)  # years
                grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
                grow_and_set(h5f["disk_Mass"], disc.Mtot())
                grow_and_set(h5f["Tc"], disc.T[0])
                grow_and_set(h5f["Sigc"], disc.Sigma[0])
                if planets is not None:
                    write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                                      chemistry_params, dust_growth_params)

        # --- once per requested snapshot time: full disc-profile row ---
        write_disc_snapshot(h5f, disc, config, t, Natom, Nmol)
        h5f.flush()


# ============================================================================
# Command-line entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run disc evolution model with HDF5 streaming output "
                    "Configuration can be loaded from a JSON file and overridden "
                    "via command-line arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_model_student.py

  # Run with custom config file
  python run_model_student.py --config config/DiscConfig_default.json

  # Override specific parameters 
  python run_model_student.py --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50

  # Use environment variable for output directory
  export DISCEVOLUTION_OUTPUT=/path/to/output
  python run_model_student.py
        """
    )
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config", "DiscConfig_default.json"),
                         help="Path to configuration JSON file")
    parser.add_argument("--psi_DW", type=float, default=None, help="Override wind parameter psi_DW")
    parser.add_argument("--Mdot", type=float, default=None, help="Override accretion rate [Msun/yr]")
    parser.add_argument("--M", type=float, default=None, help="Override disc mass [Msun]")
    parser.add_argument("--Rd", type=float, default=None, help="Override characteristic disc radius [AU]")
    parser.add_argument("--eta_abort_hours", type=float, default=None, help="Override ETA abort threshold [hours]")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {args.config}")

    overrides = {
        ("winds", "psi_DW"): args.psi_DW,
        ("disc", "Mdot"): args.Mdot,
        ("disc", "M"): args.M,
        ("disc", "Rd"): args.Rd,
        ("simulation", "eta_abort_hours"): args.eta_abort_hours,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
            print(f"Overriding {section}.{key}: {value}")

    run_model(config, cli_output_dir=args.output_dir)

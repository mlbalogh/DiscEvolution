"""
run_model_student_full.py
=========================

A compact, heavily-commented walkthrough of a single *full-physics*
DiscEvolution run, written for someone seeing this codebase for the first
time. It is the streaming-output sibling of run_model_student.py, but built
on the newer planet/planetesimal physics: it exposes the per-channel planet
accretion rates (pebble core/envelope, planetesimal in-situ, planetesimal
migration, gas) and the planetesimal eccentricity/inclination evolution.

WHAT THIS RUNS
--------------
One physical setup: a viscously- and disc-wind-accreting disc, with
two-population dust growth and radial drift, C/O chemistry, planetesimal
formation *and* planetesimal dynamical stirring, and the Bitsch-model
planet growth/migration with pebble + planetesimal + gas accretion.

UNIT CONVENTIONS (the part that trips everyone up at first)
-----------------------------------------------------------
DiscEvolution works in units where G = 1, length in AU, mass in Msun (see
DiscEvolution/constants.py). Those choices fix the time unit via Kepler's
third law -- it is NOT years. The constant `yr` (= 2*pi) converts: multiply
a duration in real years by `yr` to get code-time `t`; divide a code-time
`t` by `yr` to get real years. That is why `* yr` / `/ yr` appear so often.

    R, Rd, grid.Rc      radius                 AU
    Sigma               surface density        g / cm^2
    M (disc)            mass                   Msun
    M (planet)          core/envelope mass     Mearth
    Mdot                accretion rate         Msun / yr
    T                   temperature            K
    t (this script)     simulation time        code-time (divide by `yr`)

PIPELINE OVERVIEW
-----------------
    1. Load the JSON config (and any --flag overrides).
    2. Build grid + star + time grid.
    3. Solve the initial disc structure (disc_setup.setup_disc).
    4. Attach gas/dust transport and wrap the disc in DustGrowthTwoPop.
    5. Seed the chemistry in equilibrium with the dust.
    6. Place planets and attach the Bitsch2015Model (optional).
    7. Turn on planetesimal formation + dynamics (optional).
    8. Open the HDF5 file, create every (growable) dataset, write t = 0.
    9. Integrate forward, streaming a row per snapshot as we go.

CONFIG SCHEMA (new-physics keys this script reads)
--------------------------------------------------
    planets.planetesimal_accretion_migrate : bool
    planets.planetesimal_accretion_insitu  : bool
    planets.f_plt        : float  (embryo/planetesimal birth-mass ratio)
    planets.rho_core     : float  (protoplanet core density, g/cm^3)
    planets.Mp           : list of masses OR the strings "SI"/"PA"/"TR"
    planetesimal.rho_pltsml, .drag, .VS_embryo, .VS_pltsml, .DF,
                 .e_init, .i_init                       (dynamics switches)
    disc.d2g_SI          : streaming-instability dust fraction (SI masses)

Run it with:
    python run_model_student_full.py --config config/DiscConfig_v2.json \\
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
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolutionFV, HybridWindModel
from DiscEvolution.dust import DustGrowthTwoPop, SingleFluidDrift, PlanetesimalFormation
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.planet_formation import Planets, Bitsch2015Model
from DiscEvolution.chemistry import (
    SimpleCOChemOberg, EquilibriumCOChemOberg, TimeDepCOChemOberg, SimpleCOAtomAbund,
)

from disc_setup import setup_disc

GAS_SOLVER = ViscousEvolutionFV   # viscous scheme used when winds are off


# ============================================================================
# Step 2: time grid
# ============================================================================

def make_time_grid(sim_params):
    """
    Build the array of snapshot times (code-time units).

    sim_params['t_interval'] may be:
        "power"  -- log-spaced snapshots from t_initial to t_final (years)
        a list   -- explicit snapshot times, in Myr
        a number -- fixed linear spacing, in years
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
        return np.array(t_interval) * 1e6 * yr          # Myr -> code time

    else:
        years = np.arange(sim_params['t_initial'], sim_params['t_final'], t_interval)
        return years * yr


# ============================================================================
# Step 4: transport + dust-growth disc wrapper
# ============================================================================

def build_transport(transport_params, wind_params, disc_params, dust_growth_params, lambda_DW):
    """Build the gas/dust transport operators (any can be switched off)."""
    gas = None
    if transport_params['gas_transport']:
        gas = HybridWindModel(wind_params['psi_DW'], lambda_DW) if wind_params["on"] else GAS_SOLVER()

    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        # SingleFluidDrift folds diffusion in internally when handed a
        # `diffusion` object, so pass ours off and stop calling it separately.
        dust = SingleFluidDrift(diffusion=diffuse,
                                settling=dust_growth_params['settling'],
                                van_leer=transport_params['van_leer'])
        diffuse = None

    return gas, dust, diffuse


def build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas):
    """Wrap the bare (grid, star, eos, Sigma) disc in two-population dust growth."""
    return DustGrowthTwoPop(
        grid, star, eos, disc_params['d2g'],
        eps_SI=disc_params.get('d2g_SI', 0.02),   # SI dust fraction (only used by "SI" masses)
        Sigma=Sigma, feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
        f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
        uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas,
        rho_s=dust_growth_params.get('rho_s', 1.0),
    )


# ============================================================================
# Step 5: chemistry
# ============================================================================
#
# NOTE ON NAMING: in Tobin's original script the model name "Equilibrium"
# meant fixed C/O ratios (fix_ratios=True). Here the two cases are split
# explicitly and unambiguously: "Equilibrium" -> fix_ratios=False,
# "Equilibrium_Fixed" -> fix_ratios=True. A config written for his script
# that says "Equilibrium" should be changed to "Equilibrium_Fixed" to keep
# the same behaviour.

_CHEM_MODELS = {
    "Simple":            lambda: SimpleCOChemOberg(),
    "Equilibrium":       lambda: EquilibriumCOChemOberg(a=1e-5),
    "Equilibrium_Fixed": lambda: EquilibriumCOChemOberg(a=1e-5, fix_ratios=True),
    "TimeDep":           lambda: TimeDepCOChemOberg(a=1e-5),
}


def build_chemistry(disc, chemistry_params, d2g_target, N_cell):
    """
    Seed ice/gas abundances in equilibrium with the dust-to-gas ratio.
    Returns (chemistry_model, Nchem). Sets disc.chem and dust_frac.
    """
    if not chemistry_params["on"]:
        disc.chem = None
        return None, 0

    try:
        chemistry = _CHEM_MODELS[chemistry_params["chem_model"]]()
    except KeyError:
        raise ValueError("Valid chemistry model not selected. Choose "
                         "Simple, Equilibrium, Equilibrium_Fixed, or TimeDep")

    X_solar = SimpleCOAtomAbund(N_cell)     # solar atoms-per-H, same in every cell
    X_solar.set_solar_abundances()

    # Ice fraction and dust-to-gas ratio depend on each other, so iterate.
    chem = None
    for _ in range(100):
        if chemistry_params["assert_d2g"]:
            # Force the *total* dust-to-gas ratio to match disc.d2g exactly.
            M_dust = np.trapezoid(disc.Sigma_D.sum(0), np.pi * disc.grid.Rc ** 2)
            M_gas = np.trapezoid(disc.Sigma_G, np.pi * disc.grid.Rc ** 2)
            disc.dust_frac[:] = disc.dust_frac * (d2g_target / (M_dust / M_gas))

        chem = chemistry.equilibrium_chem(disc.T, disc.midplane_gas_density,
                                          disc.dust_frac.sum(0), X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)

    disc.chem = chem
    disc.update_ices(disc.chem.ice)
    return chemistry, disc.chem.ice.data.shape[0]


# ============================================================================
# Step 6: planets
# ============================================================================

def build_planets(disc, planet_params, chemistry_params, wind_params):
    """
    Create the Planets container + Bitsch2015Model and insert the planets.

    Returns (planets, planet_model, pending_SI). Planets whose mass is the
    string "SI" cannot be placed yet -- they need a nonzero planetesimal
    surface density -- so they are returned in `pending_SI` and inserted
    later, inside the time loop, once Sigma_planetesimal > 0 at their radius.
    """
    if not planet_params['include_planets']:
        return None, None, []

    Nchem = disc.chem.ice.data.shape[0] if chemistry_params["on"] else 0
    planets = Planets(Nchem=Nchem)

    planet_model = Bitsch2015Model(
        disc, pb_gas_f=planet_params["pb_gas_f"],
        f_plt=planet_params.get("f_plt", 400),
        migrate=planet_params["migrate"],
        pebble_acc=planet_params["pebble_accretion"],
        gas_acc=planet_params["gas_accretion"],
        planetesimal_acc_migrate=planet_params["planetesimal_accretion_migrate"],
        planetesimal_acc_insitu=planet_params["planetesimal_accretion_insitu"],
        winds=wind_params["on"],
        rho_core=planet_params.get("rho_core", 5.5),
    )
    planet_model.set_disc(disc)

    pending_SI = []
    for R_impl, M_impl, t_impl in zip(planet_params['Rp'], planet_params['Mp'],
                                      planet_params['implant_time']):
        if str(M_impl).upper() == "SI":
            pending_SI.append((t_impl, R_impl, M_impl))
        else:
            # M may be a number (Mearth) or "PA"/"TR" (birth-mass models);
            # insert_new_planet handles the string cases and sets M_env = 0.
            planet_model.insert_new_planet(t_impl, R_impl, M_impl, planets)

    return planets, planet_model, pending_SI


# ============================================================================
# Step 7: planetesimals
# ============================================================================

def build_planetesimals(disc, planets, planetesimal_params):
    """
    Attach a PlanetesimalFormation object (formation + dynamical stirring).

    `planets` is passed in because embryo viscous stirring (VS_embryo) needs
    to know the protoplanets. The drag / VS / DF switches turn the individual
    eccentricity/inclination terms on and off; set them all False to recover
    the pre-dynamics behaviour.
    """
    disc._planetesimal = None
    if not planetesimal_params['active']:
        return

    disc._planetesimal = PlanetesimalFormation(
        disc, planets,
        d_planetesimal=planetesimal_params['diameter'],
        rho_pltsml=planetesimal_params.get('rho_pltsml', 2.0),
        St_min=planetesimal_params['St_min'],
        St_max=planetesimal_params['St_max'],
        pla_eff=planetesimal_params['pla_eff'],
        drag=planetesimal_params.get('drag', True),
        VS_embryo=planetesimal_params.get('VS_embryo', True),
        VS_pltsml=planetesimal_params.get('VS_pltsml', True),
        DF=planetesimal_params.get('DF', True),
        e_init=planetesimal_params.get('e_init', 'eq'),
        i_init=planetesimal_params.get('i_init', 'eq'),
    )


# ============================================================================
# HDF5 streaming output
# ============================================================================
#
# Every quantity is a "growable" dataset: created with length 0 along axis 0
# and extended one row per snapshot with grow_and_set(). This keeps the file
# readable whether the run is 10 steps or 10 million, and keeps the dataset
# *names* stable so the analysis notebooks that key off them keep working.

def grow_and_set(dset, value):
    """Append one row to a growable HDF5 dataset."""
    n = dset.shape[0]
    dset.resize(n + 1, axis=0)
    dset[n] = value


def _ei_mechanisms(planetesimal_params):
    """The e/i stirring terms and whether each is switched on."""
    return [
        ("drag",      planetesimal_params.get('drag', True)),
        ("VS_embryo", planetesimal_params.get('VS_embryo', True)),
        ("VS_pltsml", planetesimal_params.get('VS_pltsml', True)),
        ("DF",        planetesimal_params.get('DF', True)),
    ]


def _any_ei(planetesimal_params):
    return any(flag for _, flag in _ei_mechanisms(planetesimal_params))


def output_filename(config):
    """Deterministic output filename for one parameter combination."""
    sim_params = config['simulation']
    disc_params = config['disc']
    wind_params = config['winds']
    run_name = sim_params.get('run_name', 'run')
    return (f"{run_name}_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}"
            f"_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5")


def _make_backfilled(grp, key, n_backfill):
    """Create a growable per-planet dataset, backfilled with NaN if the run
    is already underway (used when an "SI" planet is inserted late)."""
    d = grp.create_dataset(str(key), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
    if n_backfill:
        d.resize(n_backfill, axis=0)
        d[:] = np.nan
    return d


def create_planet_datasets(h5f, groups, planet_params, chemistry_params, Nchem, ip):
    """Create every per-planet dataset for planet index `ip`."""
    n_backfill = h5f["t"].shape[0]

    for name in ("Mcs", "Mes", "Rp", "disk_Mdot_p"):
        _make_backfilled(groups[name], ip, n_backfill)

    if planet_params["planetesimal_accretion_insitu"]:
        for name in ("Mdot_planetesimal", "M_iso_planetesimal"):
            _make_backfilled(groups[name], ip, n_backfill)
    if planet_params["pebble_accretion"]:
        for name in ("Mdot_pebble_core", "Mdot_pebble_env", "M_iso_pebble"):
            _make_backfilled(groups[name], ip, n_backfill)
    if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
        _make_backfilled(groups["Mdot_migration"], ip, n_backfill)
    if planet_params["gas_accretion"]:
        _make_backfilled(groups["Mdot_gas"], ip, n_backfill)

    if chemistry_params["on"]:
        pgrp_c = groups["X_cores"].create_group(str(ip))
        pgrp_e = groups["X_envs"].create_group(str(ip))
        for js in range(Nchem):
            _make_backfilled(pgrp_c, js, n_backfill)
            _make_backfilled(pgrp_e, js, n_backfill)


def create_output_file(outfile, grid, config, Nchem, alpha_SS):
    """Create the HDF5 file and every dataset/group. Returns (h5f, groups)."""
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    nR = len(grid.Rc)

    h5f = h5py.File(outfile, "w")
    h5f.attrs["alpha_SS"] = float(alpha_SS)

    # ---- scalar time series ----
    for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
        h5f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

    # ---- per-planet groups (datasets are created per planet, below) ----
    groups = {}
    if planet_params['include_planets']:
        wanted = ["Mcs", "Mes", "Rp", "disk_Mdot_p"]
        if planet_params["planetesimal_accretion_insitu"]:
            wanted += ["Mdot_planetesimal", "M_iso_planetesimal"]
        if planet_params["pebble_accretion"]:
            wanted += ["Mdot_pebble_core", "Mdot_pebble_env", "M_iso_pebble"]
        if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
            wanted += ["Mdot_migration"]
        if planet_params["gas_accretion"]:
            wanted += ["Mdot_gas"]
        if chemistry_params["on"]:
            wanted += ["X_cores", "X_envs"]
        for name in wanted:
            groups[name] = h5f.create_group(name)

    # ---- grid (written once) ----
    h5f.create_dataset("R", data=grid.Rc)

    # ---- disc-profile snapshots (one length-nR row per snapshot time) ----
    h5f.create_dataset("time_snap", shape=(0,), maxshape=(None,), dtype="f8")
    for name in ["Sigma_G", "Sigma_dust", "Sigma_pebbles",
                 "Vdrift_grains", "Vdrift_pebbles", "St_grains", "St_pebbles", "T"]:
        h5f.create_dataset(name, shape=(0, nR), maxshape=(None, nR), dtype="f8")

    if planetesimal_params['active']:
        for name in ["Sigma_planetesimals", "St_planetesimals",
                     "e_planetesimals", "i_planetesimals"]:
            h5f.create_dataset(name, shape=(0, nR), maxshape=(None, nR), dtype="f8")
        if _any_ei(planetesimal_params):
            h5f.create_dataset("de2_dt", shape=(0, nR), maxshape=(None, nR), dtype="f8")
            h5f.create_dataset("di2_dt", shape=(0, nR), maxshape=(None, nR), dtype="f8")
        for name, flag in _ei_mechanisms(planetesimal_params):
            if flag:
                h5f.create_dataset(f"de2_dt_{name}", shape=(0, nR), maxshape=(None, nR), dtype="f8")
                h5f.create_dataset(f"di2_dt_{name}", shape=(0, nR), maxshape=(None, nR), dtype="f8")

    return h5f, groups


def write_planet_row(groups, planets, planet_model, disc, grid, disk_Mdot, rates, config):
    """
    Append one row to every per-planet dataset.

    `rates` is the growth-rate dict for this instant: at t = 0 it comes from
    planet_model._growth_rates(...) (no integrate() has run yet); during the
    loop it is planet_model.rates, the rates that drove the last integrate().
    """
    planet_params = config['planets']
    chemistry_params = config['chemistry']

    for ip, planet in enumerate(planets):
        grow_and_set(groups["Mcs"][str(ip)], planet.M_core.copy())
        grow_and_set(groups["Mes"][str(ip)], planet.M_env.copy())
        grow_and_set(groups["Rp"][str(ip)], planet.R.copy())
        grow_and_set(groups["disk_Mdot_p"][str(ip)],
                     np.interp(planet.R, grid.Rc[0:-1], disk_Mdot))

        if planet_params["planetesimal_accretion_insitu"]:
            grow_and_set(groups["Mdot_planetesimal"][str(ip)],
                         rates["Mdot_planetesimal_insitu"][ip] * yr)
            grow_and_set(groups["M_iso_planetesimal"][str(ip)],
                         planet_model._pla_acc.M_iso_pltsml(planet.R))
        if planet_params["pebble_accretion"]:
            grow_and_set(groups["Mdot_pebble_core"][str(ip)], rates["Mdot_pebble_core"][ip] * yr)
            grow_and_set(groups["Mdot_pebble_env"][str(ip)], rates["Mdot_pebble_env"][ip] * yr)
            grow_and_set(groups["M_iso_pebble"][str(ip)], planet_model._peb_acc.M_iso(planet.R))
        if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
            grow_and_set(groups["Mdot_migration"][str(ip)],
                         rates["Mdot_planetesimal_migration"][ip] * yr)
        if planet_params["gas_accretion"]:
            grow_and_set(groups["Mdot_gas"][str(ip)], rates["Mdot_gas"][ip] * yr)

        if chemistry_params["on"]:
            for js, x in enumerate(planet.X_core):
                grow_and_set(groups["X_cores"][str(ip)][str(js)], x)
            for js, x in enumerate(planet.X_env):
                grow_and_set(groups["X_envs"][str(ip)][str(js)], x)


def write_disc_snapshot(h5f, disc, t, planetesimal_params):
    """Append one row to every disc-profile dataset."""
    v_drift = disc.v_drift.copy()
    stokes = disc.Stokes().copy()

    grow_and_set(h5f["time_snap"], t / (1e6 * yr))     # Myr
    grow_and_set(h5f["Sigma_G"], disc.Sigma_G)
    grow_and_set(h5f["Sigma_dust"], disc.Sigma_D[0])
    grow_and_set(h5f["Sigma_pebbles"], disc.Sigma_D[1])
    grow_and_set(h5f["Vdrift_grains"], v_drift[0])
    grow_and_set(h5f["Vdrift_pebbles"], v_drift[1])
    grow_and_set(h5f["St_grains"], stokes[0])
    grow_and_set(h5f["St_pebbles"], stokes[1])
    grow_and_set(h5f["T"], disc.T)

    if not planetesimal_params['active']:
        return

    pl = disc._planetesimal
    grow_and_set(h5f["Sigma_planetesimals"], disc.Sigma_D[2])
    grow_and_set(h5f["St_planetesimals"], stokes[2])
    grow_and_set(h5f["e_planetesimals"], pl.e)
    grow_and_set(h5f["i_planetesimals"], pl.i)

    e2, i2 = pl._e2, pl._i2
    for name, flag in _ei_mechanisms(planetesimal_params):
        if flag:
            grow_and_set(h5f[f"de2_dt_{name}"], getattr(pl, f"de2_dt_{name}")(e2, i2) * yr)
            grow_and_set(h5f[f"di2_dt_{name}"], getattr(pl, f"di2_dt_{name}")(e2, i2) * yr)
    if _any_ei(planetesimal_params):
        grow_and_set(h5f["de2_dt"], pl.de2_dt(e2, i2) * yr)
        grow_and_set(h5f["di2_dt"], pl.di2_dt(e2, i2) * yr)


# ============================================================================
# Main driver
# ============================================================================

def run_model(config, cli_output_dir=None):
    """Run one disc-evolution simulation and stream the result to HDF5."""
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

    # ---- 2. grid + star + time grid ----
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'],
                spacing=grid_params['spacing'])
    star = SimpleStar(M=star_params["M"], R=star_params["R"], T_eff=star_params['T_eff'])
    times = make_time_grid(sim_params)

    # Opacity: an instance for Tazzari; None lets IrradiatedEOS default to Zhu2012.
    kappa = Tazzari2016() if eos_params["opacity"] == "Tazzari" else None

    # ---- 3. initial disc structure (all the "solve for X" recipes live here) ----
    disc, eos, Sigma, alpha, alpha_SS, lambda_DW = setup_disc(grid, star, config, kappa)

    if alpha_SS > 5e-3:
        print(f"Not running model - alpha too high. alpha_SS={alpha_SS:.3e}, "
              f"Rd={disc_params['Rd']}, Mdisk={disc.Mtot()/Msun:.4g} Msun")
        return
    print(f"Running model. alpha_SS={alpha_SS:.3e}, Rd={disc_params['Rd']}, "
          f"Mdisk={disc.Mtot()/Msun:.4g} Msun")

    # ---- 4. transport + dust growth ----
    gas, dust, diffuse = build_transport(transport_params, wind_params, disc_params,
                                         dust_growth_params, lambda_DW)
    disc = build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas)

    # ---- 5. chemistry ----
    chemistry, Nchem = build_chemistry(disc, chemistry_params, disc_params["d2g"],
                                       grid_params["nr"])

    # ---- 6. planets, then 7. planetesimals (planets first: VS_embryo needs them) ----
    planets, planet_model, pending_SI = build_planets(disc, planet_params,
                                                      chemistry_params, wind_params)
    build_planetesimals(disc, planets, planetesimal_params)

    # ---- 8. output file + 9. integrate ----
    h5f, groups = create_output_file(outfile, grid, config, Nchem, alpha_SS)
    for ip in range(planets.N if planets is not None else 0):
        create_planet_datasets(h5f, groups, planet_params, chemistry_params, Nchem, ip)
    try:
        _integrate(h5f, groups, disc, grid, planets, planet_model, gas, dust, diffuse,
                   chemistry, times, pending_SI, Nchem, config)
    finally:
        h5f.attrs["complete"] = True
        h5f.close()
    print(f"Wrote {outfile}")


def _disc_star_mdot(disc):
    """Accretion rate onto the star at the current disc state, in Msun/yr."""
    v = disc._gas.viscous_velocity(disc, disc.Sigma)
    return -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * v * (AU * AU) * (yr / Msun)


def _growth_rates_now(planet_model, planets, chemistry_on):
    """Evaluate the planet growth rates at the current planet state (for t = 0,
    before integrate() has populated planet_model.rates)."""
    if chemistry_on:
        M_Z = (planets.X_core * planets.M_core).sum(0) + (planets.X_env * planets.M_env).sum(0)
        M_HHe = planets.M_core + planets.M_env - M_Z
    else:
        M_Z, M_HHe = planets.M_core, planets.M_env
    return planet_model._growth_rates(planets.R, planets.M_core, planets.M_env, M_Z, M_HHe)


def _integrate(h5f, groups, disc, grid, planets, planet_model, gas, dust, diffuse,
               chemistry, times, pending_SI, Nchem, config):
    """The time-stepping loop plus the periodic writes to `h5f`."""
    transport_params = config['transport']
    chemistry_params = config['chemistry']
    planet_params = config['planets']
    planetesimal_params = config['planetesimal']
    have_planets = planet_params['include_planets']

    # ---- t = 0 writes (scalars + per-planet always; disc profile only if
    #      0.0 is not itself a requested snapshot, else the loop writes it) ----
    disk_Mdot = _disc_star_mdot(disc)
    grow_and_set(h5f["t"], 0.0)
    grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
    grow_and_set(h5f["disk_Mass"], disc.Mtot())
    grow_and_set(h5f["Tc"], disc.T[0])
    grow_and_set(h5f["Sigc"], disc.Sigma[0])
    if have_planets and planets.N > 0:
        rates0 = _growth_rates_now(planet_model, planets, chemistry_params["on"])
        write_planet_row(groups, planets, planet_model, disc, grid, disk_Mdot, rates0, config)
    if 0.0 not in config['simulation']['t_interval']:
        write_disc_snapshot(h5f, disc, 0.0, planetesimal_params)
    h5f.flush()

    t, n = 0.0, 0
    for ti in times:
        while t < ti:
            # Physics-limited timestep, capped to land exactly on ti.
            dt = ti - t
            if transport_params['gas_transport']:
                dt = min(dt, disc._gas.max_timestep(disc))
            if transport_params['radial_drift']:
                dt = min(dt, dust.max_timestep(disc))

            dust_frac = getattr(disc, "dust_frac", None)
            gas_chem = disc.chem.gas.data if chemistry_params["on"] else None
            ice_chem = disc.chem.ice.data if chemistry_params["on"] else None

            # --- gas viscous/wind evolution (advects the tracers too) ---
            if transport_params['gas_transport']:
                # exclude the planetesimal band so it doesn't move with the gas
                dust_frac_gas = dust_frac[:-1] if disc._planetesimal else dust_frac
                disc._gas(dt, disc, [dust_frac_gas, gas_chem, ice_chem])

            # --- planetesimal formation + dynamical stirring ---
            if disc._planetesimal:
                disc._planetesimal.update(dt, disc, dust)

            # --- insert any pending "SI" planets now that planetesimals exist ---
            if have_planets and pending_SI and disc._planetesimal:
                still_pending = []
                for t_impl, R_impl, M_impl in pending_SI:
                    if disc.interp(R_impl, disc.Sigma_D[2]) > 0:
                        planet_model.insert_new_planet(t, R_impl, M_impl, planets)
                        create_planet_datasets(h5f, groups, planet_params,
                                               chemistry_params, Nchem, planets.N - 1)
                    else:
                        still_pending.append((t_impl, R_impl, M_impl))
                pending_SI = still_pending

            # --- dust radial drift ---
            if transport_params['radial_drift']:
                dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            # --- turbulent diffusion (only if not already folded into `dust`) ---
            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)
                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem)
                if dust_frac is not None:
                    band = dust_frac[:2] if disc._planetesimal else dust_frac[:]
                    band += dt * diffuse(disc, band)

            # --- enforce physical bounds ---
            disc.Sigma[:] = np.maximum(disc.Sigma, 0)
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
            if chemistry_params["on"]:
                disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

            # --- chemistry adsorption/desorption ---
            if chemistry_params["on"]:
                d2g = disc.dust_frac[:-1].sum(0) if disc._planetesimal else disc.dust_frac.sum(0)
                chemistry.update(dt, disc.T, disc.midplane_gas_density, d2g, disc.chem)
                disc.update_ices(disc.chem.ice)

            # --- planet growth/migration ---
            if have_planets:
                planet_model.integrate(dt, planets)

            disc.update(dt)
            t += dt
            n += 1

            if (n % 1000) == 0:
                print(f"Nstep {n} | t = {t/(1e6*yr):.4g} Myr | dt = {dt/yr:.3g} yr", flush=True)

            # --- stream scalar + per-planet series every 5 steps ---
            if (n % 5) == 0:
                disk_Mdot = _disc_star_mdot(disc)
                grow_and_set(h5f["t"], t / yr)          # years
                grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
                grow_and_set(h5f["disk_Mass"], disc.Mtot())
                grow_and_set(h5f["Tc"], disc.T[0])
                grow_and_set(h5f["Sigc"], disc.Sigma[0])
                if have_planets and planets.N > 0:
                    write_planet_row(groups, planets, planet_model, disc, grid,
                                     disk_Mdot, planet_model.rates, config)

        # --- full disc-profile row once per requested snapshot time ---
        write_disc_snapshot(h5f, disc, t, planetesimal_params)
        h5f.flush()

    if have_planets and planets.N > 0:
        h5f.create_dataset("t_form", data=planets.t_form / yr)   # yr, insertion time per planet


# ============================================================================
# Command-line entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one full-physics DiscEvolution model with HDF5 streaming output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_model_student_full.py --config config/DiscConfig_v2.json
  python run_model_student_full.py --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50
  DISCEVOLUTION_OUTPUT=/path/to/output python run_model_student_full.py
        """,
    )
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "DiscConfig_v2.json"),
                        help="Path to configuration JSON file")
    parser.add_argument("--psi_DW", type=float, default=None, help="Override wind parameter psi_DW")
    parser.add_argument("--Mdot", type=float, default=None, help="Override accretion rate [Msun/yr]")
    parser.add_argument("--M", type=float, default=None, help="Override disc mass [Msun]")
    parser.add_argument("--Rd", type=float, default=None, help="Override characteristic disc radius [AU]")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(args.config, "r") as f:
        config = json.load(f)
    print(f"Loaded configuration from: {args.config}")

    overrides = {
        ("winds", "psi_DW"): args.psi_DW,
        ("disc", "Mdot"): args.Mdot,
        ("disc", "M"): args.M,
        ("disc", "Rd"): args.Rd,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
            print(f"Overriding {section}.{key}: {value}")

    start_time = time.time()
    run_model(config, cli_output_dir=args.output_dir)
    print(f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

"""
Disc initial-condition solvers for DiscEvolution runs.

A DiscEvolution disc is set by four numbers that are all coupled through
the viscous alpha-disc relations: the viscosity parameter `alpha`, the
characteristic radius `Rd`, the disc mass `Mdisk`, and the initial
accretion rate `Mdot`. You normally know three of these from observations
or from the parameter grid you want to explore, and need to solve
(iteratively, because the equation of state depends on Sigma which depends
on the answer) for the fourth.

Each function below implements one such "fix three, solve for one" recipe.
Only the recipe actually used for the Balogh et al. HJ-formation runs is
implemented here:

    winds_alpha_disc()  -- fix Rd, Mdot, Mdisk (and the disc-wind mass-loss
                            parameter psi_DW) and solve for the alpha that
                            reproduces the target Mdot.

The original run_model_discchem_stream.py additionally implements five
more variants (fixed-Rd/solve-for-alpha without winds, solve-for-Rd,
solve-for-Mdot, the Lynden-Bell & Pringle self-similar solution, and the
wind equivalents of the Rd/Mdot solves). If you need one of those, look
there for a template and add it to `_SETUP_FUNCS` below -- the calling
script (run_model_student.py) does not need to change.
"""

import numpy as np

from DiscEvolution.constants import AU, Msun
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.viscous_evolution import HybridWindModel


def make_eos(eos_params, star, alpha_t, kappa=None, psi=None, e_rad=None):
    """
    Build the equation-of-state object named in eos_params['type'].

    alpha_t here must be the *viscous* alpha (what DiscEvolution calls
    alpha_SS): if a disc wind is present, this is smaller than the total
    alpha by a factor (1 + psi), since the wind carries away angular
    momentum without contributing to viscous heating.
    """
    eos_type = eos_params["type"]
    if eos_type == "SimpleDiscEOS":
        return SimpleDiscEOS(star, alpha_t=alpha_t)
    elif eos_type == "LocallyIsothermalEOS":
        return LocallyIsothermalEOS(star, eos_params["h0"], eos_params["q"], alpha_t)
    elif eos_type == "IrradiatedEOS":
        return IrradiatedEOS(star, alpha_t=alpha_t, kappa=kappa, psi=psi, e_rad=e_rad,
                              Tmax=eos_params["Tmax"])
    else:
        raise ValueError(f"Unknown eos type {eos_type!r}")


def winds_alpha_disc(grid, star, disc_params, eos_params, wind_params, kappa):
    """
    Solve for alpha given a fixed Rd, Mdot, Mdisk and disc-wind strength.

    Units in / out (this is the part that trips people up):
        grid.Rc                        -- radii, in AU
        disc_params['Mdot']            -- target accretion rate, in Msun/yr
        disc_params['M']               -- disc mass, in Msun (converted to
                                           grams internally, since AU/Msun
                                           are cgs constants and G = 1 in
                                           these units)
        disc_params['Rd']              -- characteristic radius, in AU
        returned Sigma                 -- surface density, in g/cm^2

    Physics:
        Sigma(R) starts as the standard self-similar power-law profile
        Sigma ~ (R/Rd)^-gamma * exp[-(R/Rd)^(2-gamma)], renormalized to
        the target disc mass. We then iterate:
            1. build the EOS (temperature structure) for the current alpha
            2. measure the Mdot that alpha actually produces
            3. rescale alpha by (target Mdot / actual Mdot)
        Steps are damped by averaging old/new alpha 50/50 each iteration,
        since the direct update can oscillate. 100 iterations converges
        comfortably for the (psi, Mdot, Mdisk, Rd) ranges used in this
        project.

        With a disc wind (psi_DW > 0) not all of alpha drives viscous
        accretion: `lambda_DW` is the wind lever-arm parameter (Tabone
        et al. 2022 notation) and `alpha_SS = alpha / (1 + psi)` is the
        viscous-only alpha that actually goes into the EOS.

    Returns
    -------
    disc, eos, Sigma, alpha, alpha_SS, lambda_DW
    """
    Mdot_target = disc_params["Mdot"]        # Msun/yr
    Mdisk = disc_params["M"] * Msun          # g
    Rd = disc_params["Rd"]                   # AU
    gamma = disc_params["gamma"]              # Sigma power-law index
    alpha = disc_params["alpha"]              # initial guess, total alpha

    psi = wind_params["psi_DW"]               # wind mass-loss parameter
    e_rad = wind_params["e_rad"]               # wind lever-arm efficiency
    if psi > 0:
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1
    else:
        # Pure-viscous limit (psi -> 0): the wind velocity v_DW in
        # HybridWindModel is proportional to psi, so it's exactly zero here
        # regardless of lambda_DW -- there's no finite limit of the lambda_DW
        # formula itself (it diverges as psi -> 0), so just pick any finite
        # placeholder value; it multiplies a wind term that's already zero.
        lambda_DW = np.inf
    alpha_SS = alpha / (1 + psi)

    R = grid.Rc
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))

    # Normalize to the requested disc mass (AccretionDisc.Mtot() integrates
    # 2*pi*R*Sigma over the grid in cgs, so this works even though `eos` is
    # not defined yet -- we only need Sigma for the mass integral here).
    disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
    Sigma *= Mdisk / disc.Mtot()

    gas = HybridWindModel(psi, lambda_DW)

    for _ in range(100):
        eos = make_eos(eos_params, star, alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
        eos.set_grid(grid)
        eos.update(0, Sigma)

        disc = AccretionDisc(grid, star, eos, Sigma)
        v_r = gas.viscous_velocity(disc, Sigma)
        Mdot_actual = disc.Mdot(v_r)[0]      # Msun/yr

        alpha = 0.5 * (alpha + alpha * Mdot_target / Mdot_actual)
        alpha_SS = alpha / (1 + psi)

    return disc, eos, Sigma, alpha, alpha_SS, lambda_DW


# Dispatch table: add an entry here (and a matching function above, or
# ported over from run_model_discchem_stream.py) to support another
# disc-initialization recipe without touching run_model_student.py.
_SETUP_FUNCS = {
    "winds-alpha": winds_alpha_disc,
}


def setup_disc(grid, star, config, kappa):
    """
    Build the initial disc for `config['grid']['type']`.

    This is the single entry point run_model_student.py calls; it just
    looks up the right recipe above and forwards the relevant config
    sections to it.
    """
    grid_type = config["grid"]["type"]
    try:
        func = _SETUP_FUNCS[grid_type]
    except KeyError:
        raise ValueError(
            f"Unknown or unsupported grid type {grid_type!r}. "
            f"This simplified module implements: {list(_SETUP_FUNCS)}. "
            "See run_model_discchem_stream.py for the other five disc-"
            "initialization recipes (Booth-alpha/Rd/Mdot, LBP, winds-Rd/"
            "winds-Mdot) if you need one of those -- port the branch over "
            "into a function here and add it to _SETUP_FUNCS."
        )
    return func(grid, star, config["disc"], config["eos"], config["winds"], kappa)

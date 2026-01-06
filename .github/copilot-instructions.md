# Copilot Instructions for DiscEvolution

## Project overview
- This package models dust+gas evolution in protoplanetary discs (viscous evolution, grain growth/drift, chemistry, planet formation, photoevaporation).
- Core code lives in the DiscEvolution/ package; example/ and control_scripts/ contain driver scripts that wire modules together via JSON configs.
- Time is usually in code units (1 year ≈ 2π), radii in AU, masses in Msun/Mearth, surface densities in Msun/AU² or g/cm² depending on context (see DiscEvolution/constants.py).

## Key architecture & data flow
- The typical model wiring pattern is: Grid → Star → EOS → Disc → Physics modules → DiscEvolutionDriver.
  - See example/run_model.py (functions setup_grid, setup_disc, setup_model, run, main) and control_scripts/run_model.py (setup_disc, setup_model, setup_output).
- Core abstractions:
  - Grid: radial mesh & cell geometry (DiscEvolution/grid.py).
  - Star: stellar parameters and Keplerian dynamics (DiscEvolution/star.py).
  - EOS: temperature & viscosity structure (DiscEvolution/eos.py: LocallyIsothermalEOS, IrradiatedEOS, SimpleDiscEOS).
  - Disc: gas/dust surface densities and derived quantities (DiscEvolution/disc.py, DiscEvolution/dust.py; e.g. AccretionDisc, DustGrowthTwoPop).
- Physics modules act on a Disc instance and are combined by DiscEvolution.driver.DiscEvolutionDriver:
  - Gas transport: DiscEvolution.viscous_evolution (ViscousEvolution, ViscousEvolutionFV, HybridWindModel, analytic LBP_Solution/TaboneSolution).
  - Dust transport: DiscEvolution.dust (SingleFluidDrift, PlanetesimalFormation, coagulation helpers).
  - Diffusion: DiscEvolution.diffusion.TracerDiffusion.
  - Chemistry: DiscEvolution.chemistry (simple C/N/O models and optional KROME wrappers).
  - Planet formation & migration: DiscEvolution.planet_formation (pebble/planetesimal/gas accretion, Type I/II migration, Bitsch2015Model) and DiscEvolution.planet.
  - Photoevaporation: DiscEvolution.photoevaporation, DiscEvolution.internal_photo, FRIED/.
- Many modules expose ASCII_header() and HDF5_attributes() used by IO/history; preserve or extend this pattern when modifying these classes so example and control scripts keep working.

## Configs, drivers, and examples
- JSON configs describe a full model setup:
  - Example: example/DiscConfig.json, referenced by example/run_model.py (DefaultModel) and parsed in main().
  - Alternative configs for photoevaporation-heavy runs live under control_scripts/ (e.g. control_scripts/DiscConfig_default.json) and in example/summer_2025/.
- New workflows should generally:
  - Reuse the existing setup_* helpers in example/run_model.py and control_scripts/run_model.py where possible.
  - Pass fully constructed modules into DiscEvolutionDriver instead of performing evolution logic directly in scripts.
- For combined disc+chemistry+planet runs, see run_model_combined_data_store.py for a modern, config-driven pattern (grid/star/EOS loop to solve for alpha, then construct dust, gas, chemistry, winds, planet formation in run_model(config)).

## Dependencies & optional components
- Base runtime dependencies are minimal and listed in requirements.txt (numpy, scipy, matplotlib). Installation is typically via pip install -e . from the repo root (uses setup.py and README.md).
- Some features are optional and guarded by imports or environment variables:
  - KROME chemistry: requires KROME_PATH and DiscEvolution.chemistry.krome_chem to be importable; example/run_model.py documents this.
  - SmoluchowskiDust / coagulation: requires an external coag_toolkit and COAG_TOOLKIT pointing to the library directory; see the try/except in example/run_model.py.
- When editing these areas, keep the "optional dependency" pattern (try/except ImportError with clear error messages) rather than hard-failing core functionality.

## Testing and typical commands
- Tests live under tests/ and use pytest-style assertions; e.g. tests/test_viscous_evo.py checks that ViscousEvolution reproduces Lynden-Bell & Pringle solutions, tests/test_io.py exercises Event_Controller.
- From the repository root, prefer:
  - python -m pytest tests
- For manual runs and debugging, common patterns are:
  - python example/run_model.py --model example/DiscConfig.json
  - python control_scripts/run_model.py (uses control_scripts/DiscConfig_default.json by default).
  - The combined pipeline in example/summer_2025/ and run_model_combined_data_store.py is used for population & chemistry studies.

## Conventions for changes
- Respect unit conventions and existing APIs: Grid.Rc in AU, star methods for Keplerian quantities, EOS for viscosity/temperature, Disc for Sigma/Sigma_D etc.
- New physics or features should be added as focused classes in DiscEvolution/ (mirroring existing modules) and then threaded into drivers or run scripts, instead of embedding complex logic directly into scripts.
- Follow existing update() and set_disc() patterns on physics modules so they can be reused after disc evolution steps (see DiscEvolution.planet_formation.GasAccretion, PebbleAccretion, PlanetesimalAccretion and migration classes).
- When touching IO or output formats, cross-check example/run_model.py and control_scripts/run_model.py to avoid breaking ASCII/HDF5 dumps and plotting/history utilities.

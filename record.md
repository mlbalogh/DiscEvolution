# Yuvan's Work

This document details all the work Yuvan has done. All commands and procedures are stored here for later reference. This is so important things aren't forgoten, to brainstorm ideas, and to have for compiling a final report. 

## Reproducing [C. Danti et al](https://www.aanda.org/articles/aa/full_html/2023/11/aa47501-23/aa47501-23.html#figs)

Command for example:

```bash
python control_scripts/run_model.py -m control_scripts/DiscConfig_replicate.json
```

Command for testing:

```python
from DiscEvolution.driver import DiscEvolutionDriver
driver = DiscEvolutionDriver(
    disc={
        "alpha"   : 1e-4,
        "mass"    : 0.128,
        "Rc"      : 137.0,
        "unit" : "jup",
        "d2g"     : 0.01,
        "Schmidt" : 1.0,
        "profile" : "LBP",
        "f_grow"  : 1.0,
        "amin"    : 0e-5
    },
    gas={
        "viscous_velocity":,

    },
    dust={
        "radii_thresholds" : [0.68, 0.95],
        "ice_frag_v" : 1000,
        "dust_frag_v" : 100,
        "p" : 3.5,
        "density" : 1.0,
        "f_grow" : 1.0 
    },
    diffusion=None, 
    chemistry={
        "on": True, 
        "type": "NoReact", 
        "fix_mu": True,
        "mu": 2.4,
        "crate": 1e-17,
        "use_abundance_file": True,
        "abundances": "Eistrup2016.dat",
        "normalize": True,
        "variable_grain_size": True
    },
    planets=[{
        "t": 1e4,
        "R": 1,
        "Mcore": 1,
        "Menv": 0
    }],
    ext_photoevaporation=None,
    int_photoevaporation=None,
    history={
        "history": True,
        "history_times": [0, 1e5, 5e5, 1e6, 3e6]
    },
    t0=0
)
driver(3e6)
```

Fixedsizedust class to start

First, replicated disk and planet configs in `control_scripts/DiscConfig_replicate.json` from [table 1](https://www.aanda.org/articles/aa/full_html/2023/11/aa47501-23/T1.html) and [table 2](https://www.aanda.org/articles/aa/full_html/2023/11/aa47501-23/T2.html).

Tested inputting lists for some parameters, testing if it works or not.

Confused why vertical alpha viscous mixing parameter is given for a 1D model. Might be important to note for later. 

Parameters not added yet, fragmentation velocity, radius of planetesimals, initial position of planetary embryos, implantation time of embryos, envelope opacity. 

## Chemistry module knowledge

To start, chemistry stores molecules in the form of strings using molecular formula notation. Ex. "H2O", "CO2", "C6H12O6".
- exception: C and Si grains. Written as "C-grain" and "Si-grain", has molecular formulas of "C" and "SiO3" respectively.

Ions are labelled with a "\+" or "\-" at the end of the string, showing their charge. Can have charges of higher values, like "\+2" or "\-5". Ex. "H\+" or "CO\+2".

Note: charge can be converted into surplus or deficency of electrons, which is written as "E" in the molecular formula. E.x. "H\-" ==> "HE". Thus, mass of electrons can be accounted for when finding molecular mass.

There are various functions that return the properties of these molecules, stored in atomic_data.py. Here's the list:
* atomic_mass(atom):
    * given an atom, returns its atomic mass in hydrogen masses.
* molecular_mass(molecule)
    * given a molecule, returns the molecular mass in hydrogen masses.
* atomic_composition(molecule, charge: bool)
    * Given a molecule, return a dictionary containing each atom in said molecule, and their number.
    * Ex. for "H2O", return {"H": 2, "O": 1}
    * If charge is true, add a "charge" term to the returned dictionary. Ex. "H\+" becomes {"H": 1, "charge": 1}
* atomic_abundances(mol_abund, charge: bool, ignore_grains: bool)
    * given molecular abundances, return atomic abundances.
    * says it returns abundance (which is percentage of specific atom compared to total atoms), but actually just returns mass or number alone, not abundance. 
    * Uses a ChemicalAbund object (simple wrapper to store data), and returns a ChemicalAbund object.

NOTE: sizes is how many data points will be taken over time. If you want to plot 3 chemical abundances, you would have size = 3, making 3 rows, thus allowing 3 entries at different times. 

Important wrappers:
* ChemicalAbund:
    * for initialize, takes list of spcies (molecules, atoms, etc.) present, then an array of the masses of given species (in order with species), and finally a "sizes" variable, which determines the number of rows in a generated 2D array. The more rows in this 2D array, the more abundance data points can be stored for species over time.
    * has many helpful functions, such as __getitem__, __setitem__, append, mass, number_abund, and more.
* MolecularIceAbund:
    * A simple wrapper for holding the fraction of species on/off a grain.
    * initialize has gas and ice objects, which are input. Not sure what exactly gas and ice is. 
    * functions are mass (get molecular mass of a specific molecule), __iter__ (iterate over species names, does not differentiate between ice and gas), and __len__ (returns int representing total number of unique species.)

Overall, 2 main chemical models in code. SimpleChemistry, and KromeChemistry. 
- Simple chemistry is built in modules that can do things like track the various species present, find abundances of given species, determine the phase (ice, gas) of species, change phase of species over time, and calculate abundances.
- Krome Chemistry is an open source package developed by T. Grassi, S. Bovino et. al to simulate chemistry specifically in astrophysics. Main website is [here](https://www.kromepackage.org) and getting started wiki is [here](https://bitbucket.org/tgrassi/krome/wiki/Get_started). Has all the capabilities of Simple Chemistry, plus chemical reactions and a much more thorough simulation (main ones are more accurate temperature calculations and factoring in cosmic rays). 

Note: in simple chemistry class, temperature is calculated separately in disc classes. Temperature determines phase of objects.

### Simple Chemistry Module

Entire module is based on the `SimpleChemBase` class. This class:
* works with C, O, and Si atomic abundances, computes molecular abundances of CO, CH4, CO2, H2O, N2, NH3, C-grains, and Si-grains.
* in initialize, sets T_cond, which is a dictionary containing condensation temperature of all molecular species listed above. 
* has 2 main functions: equilibrium_chem and update.
    * equilibrium_chem(T, rho. dust_frac, abund): computes the gas and ice molecular abundances given an abundance of particle at a given density, temperature, and dust fraction. For example, water at 1000 degrees celsius at 1 atmosphere pressure will return m_ice = m_tot, m_gas = 0. Returns info in the form of a MolecularIceAbund object.
    * update(dt, T, rho, dust_frac, chem, **kwargs): calculate new ice-gas abundances for all objects after a specified time step dt. Pretty simple phase calculations.

Another very important class that is used a lot is ThermalChem:
* Used to finds grain thermal absorption/desorption rates, used to then find gas to ice ratio.
* initialize takes number of binding sites, grain density, mean grain size by area, fraction of grain covered by binding sites, sticking probablity, and mean atomic weight in hydrogen masses.
* Has 2 main functions:
    * _equilibrium_ice_abund(T, rho, dust_frac, spec, tot_abund): finds the ice abundance at a given temperature for a given species.
    * _update_ice_balance(dt, T, rho, dust_frac, spec, abund): calculates (through thermal chemistry I don't understand) the ice and gas abundances at a given temperature, and updates relevant data points with new information for each species.
    
These 2 classes are then used for the rest of the simple chemistry module. StaticChem class uses SimpleChemBase as it's mixin class, has a function called equilibrium_ice_abund, which seems to calculate ice abundance at given T (used in equilibrium_chem for SimpleChemBase).TimeDependentChem class uses ThermalChem and SimpleChemBase. Has "update" function that uses ThermalChem's more thorough update function.EquilibriumChem also uses both SimpleChemBase and ThermalChem, can do anything those 2 classes can. Apparently "computes equilibrium of time dependent model".

Now, we start putting it all together once we mix in CNO or CO chemistry.

There are 2 wrapper classes called SimpleCNOAtomAbund and SimpleCNOMolAbund, which use ChemicalAbund and determines abundances of C, N, and O, and their relavant molecules, relative to each other. Also let's you set solar abundances.

Now, there are 2 main models of CNO chemistry, one done by Madhusudhan (2014) and another done by Oberg (2011). Both have their own classes (CNOChemOberg and CNOChemMadhu), which given temperature, gas density, dust_fraction, return molecular abundances of C-grain, Si-grain, CO2, CO, CH4, H2O, NH3, N2. The only difference is:
* CNOChemOberg: Assumes 20% of all carbon is in C-grains, and calculates the rest from there.
* CNOChemMadhu: Seems to assume that no C-grains exist in protoplanetary disk, only Si-grains and other molecules. Calculate relative molecules from there.

With everything defined, we have our overall models that get things done:
* SimpleCNOChemOberg: CNOChemOberg, StaticChem
* SimpleCNOChemMadhu: CNOChemMadhu, StaticChem
* TimeDepCNOChemOberg: CNOChemOberg, TimeDependentChem
* EquilibriumCNOChemOberg: CNOChemOberg, EquilibriumChem
* EquilibriumCNOChemMadhu: CNOChemMadhu, EquilibriumChem

The 2 classes used by all model classes handle both gas/ice phase, and abundance of each molecule. Thus, summarizing the capabilites of SimpleChemistry.

constants:
- rho: midplane gas density (g/cm^2)
- eps: dust fraction (percent of disc that is dust)
- 


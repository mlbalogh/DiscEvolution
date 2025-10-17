import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.chemistry import (
    ChemicalAbund, MolecularIceAbund, SimpleCOMolAbund, SimpleCNOAtomAbund, SimpleCNOMolAbund,
    SimpleCNOChemOberg, TimeDepCNOChemOberg, TimeDepCOChemOberg, EquilibriumCOChemMadhu,
    EquilibriumCNOChemOberg, SimpleCOAtomAbund, SimpleCOChemOberg,
    SimpleCNOChemMadhu, EquilibriumCNOChemMadhu, EquilibriumCOChemOberg
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

axes[0].set_title("Metallicity versus Psi")
axes[0].set_ylabel("Metallicity [M_earth]")
axes[0].set_xlabel("M [M_earth]")
axes[0].set_xscale("log")
axes[0].set_yscale("log")

axes[1].set_title("H2O Abundance versus Psi")
axes[1].set_ylabel("H2O Abundance [m_x / m_tot]")
axes[1].set_xlabel("M [M_earth]")
axes[1].set_xscale("log")

axes[2].set_title("C/O ratio of planet versus Psi")
axes[2].set_ylabel("$[C/O]$")
axes[2].set_xlabel("M [M_earth]")
axes[2].set_xscale("log")

axes[3].set_title("C/O ratio of envelope versus Psi")
axes[3].set_ylabel("$[C/O]_{env}$")
axes[3].set_xlabel("M [M_earth]")
axes[3].set_xscale("log")

for psi in [0, 0.01, 1, 100]:
    if psi == 0:
        psi = 0
        subset="visc"
    elif psi == 0.01:
        subset="wind"
    elif psi == 1:
        x = 75
        subset="wind"
    elif psi == 100:
        x = 100
        subset="wind"

    #fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    #axes = axes.flatten()

    #axes[0].set_title("Metallicity versus Psi")
    #axes[0].set_ylabel("Metallicity [M_earth]")
    #axes[0].set_xlabel("M [M_earth]")
    #axes[0].set_xscale("log")
    #axes[0].set_yscale("log")

    #axes[1].set_title("H2O Abundance versus Psi")
    #axes[1].set_ylabel("H2O Abundance [m_x / m_tot]")
    #axes[1].set_xlabel("M [M_earth]")
    #axes[1].set_xscale("log")

    #axes[2].set_title("C/O ratio of planet versus Psi")
    #axes[2].set_ylabel("$[C/O]$")
    #axes[2].set_xlabel("M [M_earth]")
    #axes[2].set_xscale("log")

    #axes[3].set_title("C/O ratio of envelope versus Psi")
    #axes[3].set_ylabel("$[C/O]_{env}$")
    #axes[3].set_xlabel("M [M_earth]")
    #axes[3].set_xscale("log")

    metallicity_storage = []
    H2O_abund_storage = []
    CO_ratio_storage = []
    CO_env_ratio_storage = []

    for M in [0.01, 0.05, 0.1]:
        if M == 0.01:
            size = 100
        elif M == 0.05:
            size = 200
        elif M == 0.1:
            size = 300

        for Mdot in [1e-8, 1e-9]:
            if Mdot == 1e-8:
                color = "green"
            elif Mdot == 1e-9:
                color = "blue"
            
            for Rd in [10, 50, 100]:
                if Rd == 10:
                    marker = "^"
                    #offset = 5
                elif Rd == 50:
                    marker = "s"
                    #offset = 0
                elif Rd == 100:
                    marker = "o"
                    #offset = -5

                # exclude discs with unphysical alpha
                # slap on fix, remove once new data is through with attached alpha
                if psi < 100:
                    if Mdot == 1e-8:
                        if M == 0.01:
                            if Rd == 100:
                                continue
                    elif Mdot == 1e-7:
                        if M == 0.1:
                            if Rd == 100:
                                continue
                        elif M == 0.01:
                            continue

                with open(f"yuvan_data/{subset}/{subset}s_mig_psi{psi}_Mdot{Mdot:.1e}_M{M:.1e}_Rd{Rd:.1e}.json", 'r') as i:
                    visc_data = json.load(i)

                    t = visc_data["t"]
                    Mcs = visc_data["Mcs"]
                    Mes = visc_data["Mes"]
                    Rp = visc_data["Rp"]
                    X_cores = visc_data["X_cores"]
                    X_envs = visc_data["X_envs"]

                    total_mass = []
                    metallicity = []
                    H2O_abund = []
                    CO_ratio = []
                    CO_env_ratio = []
                    for i, core_mass in enumerate(Mcs):
                        if core_mass[-1] + Mes[i][-1] <= 20:
                            continue # excluding non-gas giants
                        else:
                            # total mass for metallicity
                            total_mass.append(float(core_mass[-1]) + float(Mes[i][-1]))

                        planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
                        planetary_mol_abund.data[:] = ((np.array(X_cores[i]))*np.array(Mcs[i]) + np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]
                        plan_env_mol_abund = SimpleCOMolAbund(len(X_envs[0][0]))
                        plan_env_mol_abund.data[:] = (np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]

                        # collect metallicity and H2O abund data
                        metallicity.append(((np.array(X_cores[i]).sum(0)[-1])*np.array(Mcs[i][-1]) + np.array(X_envs[i]).sum(0)[-1]*np.array(Mes[i][-1]))/total_mass[-1])
                        H2O_abund.append(planetary_mol_abund["H2O"][-1])

                        # now find C/O in envelope and as a whole
                        planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                        CO_ratio.append(planetary_atom_abund.number_abund("C")[-1]/planetary_atom_abund.number_abund("O")[-1])

                        plan_env_atom_abund = plan_env_mol_abund.atomic_abundance()
                        CO_env_ratio.append(plan_env_atom_abund.number_abund("C")[-1]/plan_env_atom_abund.number_abund("O")[-1])

                        if any(np.array(metallicity) == float('inf')) or any(np.array(metallicity) == float('-inf')):
                            metallicity.pop(-1)
                            H2O_abund.pop(-1)
                            CO_ratio.pop(-1)
                            CO_env_ratio.pop(-1)
                            total_mass.pop(-1)
                            continue

                    if len(metallicity) != 0:
                        metallicity_storage += metallicity
                        H2O_abund_storage += H2O_abund
                        CO_ratio_storage += CO_ratio
                        CO_env_ratio_storage += CO_env_ratio
                    
                        axes[0].scatter(total_mass, np.array(metallicity)*np.array(total_mass), color=color, s=size, marker=marker)
                        axes[1].scatter(total_mass, H2O_abund, color=color, s=size, marker=marker)
                        axes[2].scatter(total_mass, CO_ratio, color=color, s=size, marker=marker)
                        axes[3].scatter(total_mass, CO_env_ratio, color=color, s=size, marker=marker)

    #axes[0].scatter(x, sum(metallicity_storage)/len(metallicity_storage), zorder=3, color="red", s=150)
    #axes[1].scatter(x, sum(H2O_abund_storage)/len(H2O_abund_storage), zorder=3, color="red", s=150)
    #axes[2].scatter(x, sum(CO_ratio_storage)/len(CO_ratio_storage), zorder=3, color="red", s=150)
    #axes[3].scatter(x, sum(CO_env_ratio_storage)/len(CO_env_ratio_storage), zorder=3, color="red", s=150)

    #plt.tight_layout()
    #fig.savefig(f"graphs/results/M_vs_chem_all_plans_psi{psi}.png", bbox_inches="tight")
    #plt.clf()

#axes[0].scatter(x, sum(metallicity_storage)/len(metallicity_storage), zorder=3, color="red", s=150)
#axes[1].scatter(x, sum(H2O_abund_storage)/len(H2O_abund_storage), zorder=3, color="red", s=150)
#axes[2].scatter(x, sum(CO_ratio_storage)/len(CO_ratio_storage), zorder=3, color="red", s=150)
#axes[3].scatter(x, sum(CO_env_ratio_storage)/len(CO_env_ratio_storage), zorder=3, color="red", s=150)

plt.tight_layout()
fig.savefig(f"graphs/results/initial_run_M_vs_chem_gas_giants.png", bbox_inches="tight")
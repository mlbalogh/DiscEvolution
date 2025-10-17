import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.chemistry import *
import random

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

axes[0].set_title("Final Metallicity versus Psi")
axes[0].set_ylabel("Metallicity [M_earth]")
axes[0].set_xlabel("Final Planet Radius [AU]")
axes[0].set_yscale("log")
axes[0].set_xscale("log")
#axes[0].set_ylim(1e0, 5e2)

axes[1].set_title("Final Volatile to Refractory Ratio versus Psi")
axes[1].set_ylabel("X_vol/X_refrac")
axes[1].set_xlabel("Final Planet Radius [AU]")
axes[1].set_xscale("log")

axes[2].set_title("Final C/O ratio of planet versus Psi")
axes[2].set_ylabel("$[C/O]$")
axes[2].set_xlabel("Final Planet Radius [AU]")
axes[2].set_xscale("log")

axes[3].set_title("Final C/O ratio of envelope versus Psi")
axes[3].set_ylabel("$[C/O]_{env}$")
axes[3].set_xlabel("Final Planet Radius [AU]")
axes[3].set_xscale("log")

Mdot = 1e-7
color_store = []

for psi in [0.01, 1, 1000]:
    if psi == 0.01:
        subset="wind"
        color = "red"
    elif psi == 1:
        subset="wind"
        color = "blue"
    elif psi == 1000:
        subset="wind"
        color = "green"

    metallicity_storage = []
    vol_to_refrac_storage = []
    CO_ratio_storage = []
    CO_env_ratio_storage = []

    for M in [0.01, 0.05, 0.1]:
        if M == 0.01:
            size = 100
        elif M == 0.05:
            size = 200
        elif M == 0.1:
            size = 300
            
        for Rd in [10, 50, 100]:
            if Rd == 10:
                marker = "^"
            elif Rd == 50:
                marker = "s"
            elif Rd == 100:
                marker = "o"

            try:
                open(f"data/wind/winds_mig_psi{psi}_Mdot{Mdot:.1e}_M{M:.1e}_Rd{Rd:.1e}.json", 'r')
            except FileNotFoundError:
                continue

            with open(f"data/wind/winds_mig_psi{psi}_Mdot{Mdot:.1e}_M{M:.1e}_Rd{Rd:.1e}.json", 'r') as i:
                visc_data = json.load(i)

                t = visc_data["t"]
                Mcs = visc_data["Mcs"]
                Mes = visc_data["Mes"]
                Rp = visc_data["Rp"]
                X_cores = visc_data["X_cores"]
                X_envs = visc_data["X_envs"]

                total_mass = []
                metallicity = []
                vol_to_refrac_ratio = []
                CO_ratio = []
                CO_env_ratio = []
                end_radius = []
                for i, core_mass in enumerate(Mcs):
                    total_mass.append(float(core_mass[-1]) + float(Mes[i][-1]))
                    end_radius.append(Rp[i][-1])

                    planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
                    planetary_mol_abund.data[:] = ((np.array(X_cores[i]))*np.array(Mcs[i]) + np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]
                    plan_env_mol_abund = SimpleCOMolAbund(len(X_envs[0][0]))
                    plan_env_mol_abund.data[:] = (np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]

                    vol_to_refrac_ratio.append(planetary_mol_abund.data[:4].sum(0)[-1]/planetary_mol_abund.data[4:].sum(0)[-1])

                    # collect metallicity and H2O abund data
                    metallicity.append(((np.array(X_cores[i]).sum(0)[-1])*np.array(Mcs[i][-1]) + np.array(X_envs[i]).sum(0)[-1]*np.array(Mes[i][-1]))/total_mass[-1])

                    # now find C/O in envelope and as a whole
                    planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                    CO_ratio.append(planetary_atom_abund.number_abund("C")[-1]/planetary_atom_abund.number_abund("O")[-1])

                    plan_env_atom_abund = plan_env_mol_abund.atomic_abundance()
                    CO_env_ratio.append(plan_env_atom_abund.number_abund("C")[-1]/plan_env_atom_abund.number_abund("O")[-1])

                    if any(np.array(metallicity) == float('inf')) or any(np.array(metallicity) == float('-inf')):
                        metallicity.pop(-1)
                        vol_to_refrac_ratio.pop(-1)
                        CO_ratio.pop(-1)
                        CO_env_ratio.pop(-1)
                        end_radius.pop(-1)
                        total_mass.pop(-1)
                        continue

                if len(metallicity) != 0:
                    metallicity_storage += metallicity
                    vol_to_refrac_storage += vol_to_refrac_ratio
                    CO_ratio_storage += CO_ratio
                    CO_env_ratio_storage += CO_env_ratio
                    
                    l1 = axes[0].scatter(np.array(end_radius), np.array(metallicity)*np.array(total_mass), color=color, s=size, marker=marker)
                    axes[1].scatter(np.array(end_radius), vol_to_refrac_ratio, color=color, s=size, marker=marker)
                    axes[2].scatter(np.array(end_radius), CO_ratio, color=color, s=size, marker=marker)
                    axes[3].scatter(np.array(end_radius), CO_env_ratio, color=color, s=size, marker=marker)

    color_store.append(l1)

axes[0].legend([item for item in color_store], ["psi = 0.01", "psi = 1", "psi = 1000"], loc="lower right")

plt.tight_layout()
fig.savefig(f"graphs/results/idk_3_Mdot{Mdot:.0e}.png", bbox_inches="tight")
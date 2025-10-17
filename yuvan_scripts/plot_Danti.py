import json
import numpy as np
import matplotlib.pyplot as plt
from DiscEvolution.constants import *
from DiscEvolution.chemistry import *

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes = axes.flatten()

Mdot = 1e-8
color_store = []
numb = 0

for psi in [0.01, 1, 1000]:

    if psi == 0:
        subset = "visc"
        color="black"
    elif psi == 0.01:
        subset="wind"
        color = "red"
    elif psi == 1:
        subset="wind"
        color = "blue"
    elif psi == 1000:
        subset="wind"
        color = "green"

    axes[numb].set_title(f"Metallicity versus Mass for Psi {psi:.0e}")
    axes[numb].set_ylabel("Metallicity [M/M_earth]")
    axes[numb].set_xlabel("Mass [M/M_jup]")
    axes[numb].set_xscale("log")
    axes[numb].set_yscale("log")
    axes[numb].set_xlim(1e-2, 1e2)
    axes[numb].set_ylim(1e0, 1e3)

    metallicity_storage = []

    for M in [0.01, 0.05, 0.1]:
        if M == 0.01:
            size = 10
        elif M == 0.05:
            size = 20
        elif M == 0.1:
            size = 30
            
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

            try:
                open(f"data/{subset}/{subset}s_mig_psi{psi}_Mdot{Mdot:.1e}_M{M:.1e}_Rd{Rd:.1e}.json", 'r')
            except FileNotFoundError:
                continue

            with open(f"data/{subset}/{subset}s_mig_psi{psi}_Mdot{Mdot:.1e}_M{M:.1e}_Rd{Rd:.1e}.json", 'r') as i:
                visc_data = json.load(i)

                t = visc_data["t"]
                Mcs = visc_data["Mcs"]
                Mes = visc_data["Mes"]
                Rp = visc_data["Rp"]
                X_cores = visc_data["X_cores"]
                X_envs = visc_data["X_envs"]

                total_mass = []
                metallicity = []

                for i, core_mass in enumerate(Mcs):
                    #if core_mass[-1] + Mes[i][-1] <= 20:
                        #continue # excluding non-gas giants
                    #else:
                        # total mass for metallicity
                    total_mass.append(float(core_mass[-1]) + float(Mes[i][-1]))

                    planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
                    planetary_mol_abund.data[:] = ((np.array(X_cores[i]))*np.array(Mcs[i]) + np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]
                    plan_env_mol_abund = SimpleCOMolAbund(len(X_envs[0][0]))
                    plan_env_mol_abund.data[:] = (np.array(X_envs[i])*np.array(Mes[i]))/total_mass[-1]

                    # collect metallicity and H2O abund data
                    metallicity.append(((np.array(X_cores[i]).sum(0)[-1])*np.array(Mcs[i][-1]) + np.array(X_envs[i]).sum(0)[-1]*np.array(Mes[i][-1]))/total_mass[-1])

                    if any(np.array(metallicity) == float('inf')) or any(np.array(metallicity) == float('-inf')):
                        metallicity.pop(-1)
                        continue

                if len(metallicity) != 0:
                    metallicity_storage += metallicity
                    
                    l1 = axes[numb].scatter(np.array(total_mass)*Mearth/Mjup, np.array(metallicity)*np.array(total_mass), color=color, s=size, marker=marker)

    color_store.append(l1)

    #axes[0].scatter(x, sum(metallicity_storage)/len(metallicity_storage), zorder=3, color="red", s=150)
    #axes[1].scatter(x, sum(H2O_abund_storage)/len(H2O_abund_storage), zorder=3, color="red", s=150)
    #axes[2].scatter(x, sum(CO_ratio_storage)/len(CO_ratio_storage), zorder=3, color="red", s=150)
    #axes[3].scatter(x, sum(CO_env_ratio_storage)/len(CO_env_ratio_storage), zorder=3, color="red", s=150)

    #plt.tight_layout()
    #fig.savefig(f"graphs/results/M_vs_chem_all_plans_psi{psi}.png", bbox_inches="tight")
    #plt.clf()

    numb+=1

#axes[0].scatter(x, sum(metallicity_storage)/len(metallicity_storage), zorder=3, color="red", s=150)
#axes[1].scatter(x, sum(H2O_abund_storage)/len(H2O_abund_storage), zorder=3, color="red", s=150)
#axes[2].scatter(x, sum(CO_ratio_storage)/len(CO_ratio_storage), zorder=3, color="red", s=150)
#axes[3].scatter(x, sum(CO_env_ratio_storage)/len(CO_env_ratio_storage), zorder=3, color="red", s=150)

plt.tight_layout()
fig.savefig(f"graphs/results/Danti_repro_Mdot{Mdot:.0e}.png", bbox_inches="tight")
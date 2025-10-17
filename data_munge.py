import csv
import numpy as np
import json
from DiscEvolution.constants import *
from DiscEvolution.chemistry import *

data = [["ID", "Disk Mdot", "Disk Psi", "Disk Mass", "Disk Rd", "Time (yr)", "Radius (AU)", "Core Mass (M_earth)", "Envelope Mass (M_earth)", "Planetary Metallicity (M_earth)", "Vol/Ref", "C/O ratio", "C/O envelope ratio", "Water Ratio", "O/H ratio", "O/H envelope ratio", "C/H ratio", "C/H envelope ratio"]]
ID = 1

for psi in [0.01, 1, 1000]:

    for M in [0.01, 0.05, 0.1]:
            
        for Rd in [10, 50, 100]:

            for Mdot in [float(1e-8), float(1e-9), float(1e-7)]:

                disk_params = {"psi": psi, "M": M, "Rd": Rd, "Mdot": Mdot}

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

                    for i, core_mass in enumerate(Mcs):
                        total_mass = np.array(core_mass) + np.array(Mes[i])

                        planetary_mol_abund = SimpleCOMolAbund(len(X_cores[0][0]))
                        planetary_mol_abund.data[:] = ((np.array(X_cores[i]))*np.array(Mcs[i]) + np.array(X_envs[i])*np.array(Mes[i]))/total_mass
                        plan_env_mol_abund = SimpleCOMolAbund(len(X_envs[0][0]))
                        plan_env_mol_abund.data[:] = np.array(X_envs[i])

                        vol_to_refrac_ratio = planetary_mol_abund.data[:4].sum(0)/planetary_mol_abund.data[4:].sum(0)

                        # collect metallicity and H2O abund data
                        metallicity = ((np.array(X_cores[i]).sum(0))*np.array(Mcs[i]) + np.array(X_envs[i]).sum(0)*np.array(Mes[i]))/total_mass

                        # now find C/O in envelope and as a whole
                        planetary_atom_abund = planetary_mol_abund.atomic_abundance()
                        CO_ratio = np.nan_to_num(planetary_atom_abund["C"]/planetary_atom_abund["O"])

                        plan_env_atom_abund = plan_env_mol_abund.atomic_abundance()
                        CO_env_ratio = np.nan_to_num(plan_env_atom_abund["C"]/plan_env_atom_abund["O"])

                        water_fraction = planetary_mol_abund["H2O"]

                        H_abund = 2*planetary_mol_abund["H2O"]/planetary_mol_abund.mass("H2O") + 4*planetary_mol_abund["CH4"]/planetary_mol_abund.mass("CH4") + 2*(1-planetary_mol_abund.total_abund)/2
                        H_env_abund = 2*plan_env_mol_abund["H2O"]/plan_env_mol_abund.mass("H2O") + 4*plan_env_mol_abund["CH4"]/plan_env_mol_abund.mass("CH4") + 2*(1-plan_env_mol_abund.total_abund)/2

                        OH_ratio = planetary_atom_abund["O"]/H_abund
                        OH_env_ratio = plan_env_atom_abund["O"]/H_env_abund

                        CH_ratio = planetary_atom_abund["C"]/H_abund
                        CH_env_ratio = plan_env_atom_abund["C"]/H_env_abund

                        while t[-1] == t[-2]:
                            t.pop(-1)

                        if any(np.array(metallicity) == float('inf')) or any(np.array(metallicity) == float('-inf')):
                            continue
                    
                        for num, item in enumerate(total_mass):
                            Mcs[i][num] = float(Mcs[i][num])
                            Mes[i][num] = float(Mes[i][num])
                            Rp[i][num] = float(Rp[i][num])
                            metallicity[num] = float(metallicity[num])
                            vol_to_refrac_ratio[num] = float(vol_to_refrac_ratio[num])
                            CO_ratio[num] = float(CO_ratio[num])
                            CO_env_ratio[num] = float(CO_env_ratio[num])
                            water_fraction[num] = float(water_fraction[num])
                            OH_ratio[num] = float(OH_ratio[num])
                            OH_env_ratio[num] = float(OH_env_ratio[num])
                            CH_ratio[num] = float(CH_ratio[num])
                            CH_env_ratio[num] = float(CH_env_ratio[num])
                        
                        data.append([ID, float(Mdot), float(psi), float(M), float(Rd), t, list(Rp[i]), list(Mcs[i]), list(Mes[i]), list(metallicity), list(vol_to_refrac_ratio), list(CO_ratio), list(CO_env_ratio), list(water_fraction), list(OH_ratio), list(OH_env_ratio), list(CH_ratio), list(CH_env_ratio)])
                        ID += 1

with open("data/compressed_data.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)
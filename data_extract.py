import csv
import numpy as np
from DiscEvolution.constants import *
from DiscEvolution.chemistry import *

with open("data/compressed_data.csv", "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    
    header = next(csv_reader)
    
    IDs = np.array([])
    radius = []
    mass = []
    time = []
    metallicity = []
    vol_to_refrac = []
    CO_ratio = []
    CO_env_ratio = []
    water_frac = []
    Mdot = np.array([])
    M = np.array([])
    psi = np.array([])
    Rd = np.array([])

    for planet in csv_reader:
        IDs = np.append(IDs, int(planet[0]))
        Mdot = np.append(Mdot, float(planet[1]))
        psi = np.append(psi, float(planet[2]))
        M = np.append(M, float(planet[3]))
        Rd = np.append(Rd, float(planet[4]))

        temp_data_store = []

        index = 5
        while index < len(planet):
            planet[index] = planet[index].replace("[", "").replace("]", "").replace(" ", "").replace("np.float64(", "").replace(")", "")
            temp_data_store.append([float(item) for item in planet[index].split(",")])
            index+=1

        time.append(np.array(temp_data_store[0]))
        radius.append(np.array(temp_data_store[1]))
        mass.append(np.array(temp_data_store[2]))
        metallicity.append(np.array(temp_data_store[3]))
        vol_to_refrac.append(np.array(temp_data_store[4]))
        CO_ratio.append(np.array(temp_data_store[5]))
        CO_env_ratio.append(np.array(temp_data_store[6]))
        water_frac.append(np.array(temp_data_store[7]))
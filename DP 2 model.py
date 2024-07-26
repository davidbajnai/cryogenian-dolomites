# This code is used to:
# Model seawater oxygen isotope compositions based on the extended Muehlenbachs model

# INPUT:  DP Table S2.csv (carbonate data)

# OUTPUT: DP Table S3.csv (modelled seawater compositions)
#         DP Table S4.csv (best-fit compositions)

# >>>>>>>>>

# Import libraries
import sys
import os
import numpy as np
from scipy.optimize import fsolve
import warnings
from tqdm import tqdm
import pandas as pd

# Import functions
from functions import *

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# Define additional functions
def a18c(T):

    # Daeron et al. (2019) - calcite
    # return np.exp((17.57 * 1000 / T - 29.13) / 1000)

    # Wostbrock et al. (2020) - calcite
    # return np.exp((2.45*10**6/((T)**2)+0.49/(T))/1000)

    # Guo and Zhou (2019) – aragonite
    return 0.0201 * (1000 / T) + 0.9642

    # Hayles et al. (2018) - calcite
    # B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / T**4 + -4.495755E+05 / T**3  + 1.307870E+04 / T**2 + -5.393675E-01 / T + 1.331245E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_calcite) / np.exp(B_water)


def theta_c(T):

    # Wostbrock et al. (2020) - calcite
    # return -1.39 / T + 0.5305

    # Guo and Zhou (2019) – aragonite
    return 59.1047/T**2 + -1.4089/T + 0.5297

    # Hayles et al. (2018) - calcite
    # K_calcite = 1.019124E+09 / T**5 + -2.117501E+07 / T**4 + 1.686453E+05 / \
    #     T**3 + -5.784679E+02 / T**2 + 1.489666E-01 / T + 0.5304852
    # B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / \
    #     T**4 + -4.495755E+05 / T**3 + 1.307870E+04 / \
    #     T**2 + -5.393675E-01 / T + 1.331245E-04
    # K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / \
    #     T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / \
    #     T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # a18 = np.exp(B_calcite) / np.exp(B_water)
    # return K_calcite + (K_calcite-K_water) * (B_water / np.log(a18))


def a17c(T):
    return a18c(T) ** theta_c(T)


def a18qz(T):
    return np.exp((4.2 * 10**6 / T**2 - 3.3 * 10**3 / T) / 1000)


def theta_qz(T):
    return -1.85 / T + 0.5305


def a17qz(T):
    return a18qz(T)**theta_qz(T)


def d18Oqz(equilibrium_temperatures, d18Ow):
    return a18qz(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d17Oqz(equilibrium_temperatures, d18Ow):
    return a17qz(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d18Oc(equilibrium_temperatures, d18Ow):
    return a18c(equilibrium_temperatures) * (d18Ow+1000) - 1000


def d17Oc(equilibrium_temperatures, d18Ow):
    return a17c(equilibrium_temperatures) * (d18Ow+1000) - 1000


def D_carbonate_seawater(temp):
    temp = temp + 273.15
    Dd18O_carbonate_seawater = 1000*(a18c(temp)-1)
    Dd17O_carbonate_seawater = 1000*(a17c(temp)-1)
    return Dd18O_carbonate_seawater, Dd17O_carbonate_seawater


def D_silicate_seawater(temp):
    temp = temp + 273.15
    Dd18O_silicate_seawater = 1000*(a18qz(temp)-1)
    Dd17O_silicate_seawater = 1000*(a17qz(temp)-1)
    return Dd18O_silicate_seawater, Dd17O_silicate_seawater


def calculate_normalised_fluxes(dictionary):
    M_ocean = 1.4e24 * (16/18.015) # grams of oxygen in the ocean
    F17O = ((dictionary["flux"] * 1e15 * dictionary["factor"]) / M_ocean * dictionary["Dd17O"])*10e9
    F18O = ((dictionary["flux"] * 1e15 * dictionary["factor"]) / M_ocean * dictionary["Dd18O"])*10e9
    return F17O, F18O


def calculate_total_flux(d18O_sw, d17O_sw, fF_sp, fF_sfw, fF_cw, fF_cg, fF_r, fF_c, fF_qz, cT, qzT):

    d18O_MORB = 5.8    # Sengupta and Pack (2018)
    Dp17O_MORB = -46   # Sengupta and Pack (2018)
    d17O_MORB = d17O(d18O_MORB, Dp17O_MORB)

    d18O_cont = 7      # Sengupta and Pack (2018)
    Dp17O_cont = -43   # Sengupta and Pack (2018)
    d17O_cont = d17O(d18O_cont, Dp17O_cont)

    d18O_meta = 12     # Sengupta and Pack (2018)
    Dp17O_meta = -48   # Sengupta and Pack (2018)
    d17O_meta = d17O(d18O_meta, Dp17O_meta)

    Dd18O_met = -3     # Herwartz et al. (2021)
    DDp17O_met = 11    # Herwartz et al. (2021)
    Dd17O_met = d17O(Dd18O_met, DDp17O_met)

    Dd18O_clay = 20     # Herwartz et al. (2021)
    DDp17O_clay = -126  # Herwartz et al. (2021)
    Dd17O_clay = d17O(Dd18O_clay, DDp17O_clay)

    Dd18O_hydro = 4.5   # Sengupta and Pack (2018)
    DDp17O_hydro = -8   # Sengupta and Pack (2018)
    Dd17O_hydro = d17O(Dd18O_hydro, DDp17O_hydro)

    Dd18O_seaf = 25     # Sengupta and Pack (2018)
    DDp17O_seaf = -83   # Sengupta and Pack (2018)
    Dd17O_seaf = d17O(Dd18O_seaf, DDp17O_seaf)

    Dd18O_sed = 26      # Sengupta and Pack (2018)
    DDp17O_sed = -81    # Sengupta and Pack (2018)
    Dd17O_sed = d17O(Dd18O_sed, DDp17O_sed)

    Dd18O_Connate_water_seawater = 3     # Herwartz et al. (2021)   
    DDp17O_Connate_water_seawater = -51  # Herwartz et al. (2021)      
    Dd17O_Connate_water_seawater = d17O(Dd18O_Connate_water_seawater, DDp17O_Connate_water_seawater)

    Dd18O_carbonate_seawater, Dd17O_carbonate_seawater = D_carbonate_seawater(cT)
    Dd18O_silicate_seawater, Dd17O_silicate_seawater = D_silicate_seawater(qzT)

    # High temperature alteration of oceanic crust
    global F_sp
    F_sp = {
        "name": "High temperature alteration of oceanic crust",
        "abbreviation": r"F$_{sp}$",
        "flux": 18.3,   # Muehlenbachs et al. (1998), Table 1
        "factor": fF_sp
    }
    F_sp["Dd18O"] = d18O_MORB - (d18O_sw+Dd18O_hydro)
    F_sp["Dd17O"] = d17O_MORB - (d17O_sw+Dd17O_hydro)
    F_sp["F17O"], F_sp["F18O"] = calculate_normalised_fluxes(F_sp)

    # Sea floor weathering
    global F_sfw
    F_sfw = {
        "name": "Low temperature alteration of oceanic crust",
        "abbreviation": r"F$_{sfw}$",
        "flux": 2.2,    # Muehlenbachs et al. (1998), Table 1
        "factor": fF_sfw
    }
    F_sfw["Dd18O"] = d18O_MORB - (0.2*(d18O_sw+Dd18O_seaf)+0.8*d18O_MORB)
    F_sfw["Dd17O"] = d17O_MORB - (0.2*(d17O_sw+Dd17O_seaf)+0.8*d17O_MORB)
    F_sfw["F17O"], F_sfw["F18O"] = calculate_normalised_fluxes(F_sfw)

    # Continental weathering
    global F_cw
    F_cw = {
        "name": "Continental weathering",
        "abbreviation": r"F$_{cw}$",
        "flux": 10,    # Muehlenbachs et al. (1998), Table 1
        "factor":  fF_cw
    }
    F_cw["Dd18O"] = -0.125*(2*(d18O_sw + Dd18O_clay + Dd18O_met)-d18O_cont-d18O_meta)
    F_cw["Dd17O"] = -0.125*(2*(d17O_sw + Dd17O_clay + Dd17O_met)-d17O_cont-d17O_meta)
    F_cw["F17O"], F_cw["F18O"] = calculate_normalised_fluxes(F_cw)

    # Continental growth
    global F_cg
    F_cg = {
        "name": "Continental growth",
        "abbreviation": r"F$_{cg}$",
        "flux": 1.5,    # Muehlenbachs et al. (1998), Table 1
        "factor": fF_cg
    }
    F_cg["Dd18O"] = d18O_MORB-(0.1*(d18O_sw + Dd18O_sed)+0.9*d18O_MORB)
    F_cg["Dd17O"] = d17O_MORB-(0.1*(d17O_sw + Dd17O_sed)+0.9*d17O_MORB)
    F_cg["F17O"], F_cg["F18O"] = calculate_normalised_fluxes(F_cg)

    # Mantle recycling of water
    global F_r
    F_r = {
        "name": "Mantle recycling of water",
        "abbreviation": r"F$_{r}$",
        "flux": 0.8,    # Muehlenbachs et al. (1998), Table 1
        "factor": fF_r
    }
    F_r["Dd18O"] = d18O_MORB-(d18O_sw+Dd18O_Connate_water_seawater)
    F_r["Dd17O"] = d17O_MORB-(d17O_sw+Dd17O_Connate_water_seawater)
    F_r["F17O"], F_r["F18O"] = calculate_normalised_fluxes(F_r)

    # Carbonatization
    global F_c
    F_c = {
        "name": "Carbonatisation",
        "abbreviation": r"F$_{CO2}$",
        "flux": 0.0768,   # Alt and Teagle (1999)
        "factor": fF_c
    }
    F_c["Dd18O"] = d18O_MORB-(d18O_sw+Dd18O_carbonate_seawater)
    F_c["Dd17O"] = d17O_MORB-(d17O_sw+Dd17O_carbonate_seawater)
    F_c["F17O"], F_c["F18O"] = calculate_normalised_fluxes(F_c)

    # Silicification
    global F_qz
    F_qz = {
        "name": "Silicification",
        "abbreviation": r"F$_{SiO2}$",
        "flux": 0.0768,   # assumed to be the same as carbonatization
        "factor": fF_qz
    }
    F_qz["Dd18O"] = d18O_MORB-(d18O_sw+Dd18O_silicate_seawater)
    F_qz["Dd17O"] = d17O_MORB-(d17O_sw+Dd17O_silicate_seawater)
    F_qz["F17O"], F_qz["F18O"] = calculate_normalised_fluxes(F_qz)

    # calculate the total flux
    F18_total = F_sp["F18O"] + F_sfw["F18O"] + F_cw["F18O"] + \
        F_cg["F18O"] + F_r["F18O"] + \
        F_c["F18O"] + F_qz["F18O"]
    F17_total = F_sp["F17O"] + F_sfw["F17O"] + F_cw["F17O"] + \
        F_cg["F17O"] + F_r["F17O"] + \
        F_c["F17O"] + F_qz["F17O"]

    return F17_total, F18_total

# Function to be passed to fsolve
def equations_to_solve(d_seawater, *args):
    d18O_sw, d17O_sw = d_seawater
    return calculate_total_flux(d18O_sw, d17O_sw, *args)

# Define the Monte Carlo simulation function
def monte_carlo_simulation(num_simulations):
    results = []
    for _ in tqdm(range(num_simulations)):
        
        # Generate random parameters
        fF_sp = np.random.uniform(0.5, 1)
        fF_sfw = np.random.uniform(1, 5)
        fF_cw = np.random.uniform(1, 5)
        fF_cg = np.random.uniform(0.9, 1.1)
        fF_r = np.random.uniform(0.9, 1.1)
        fF_c = np.random.uniform(0, 110)
        fF_qz = fF_c
        cT = np.random.uniform(100, 200)
        qzT = np.random.uniform(100, 200)
        
        # Initial guess for d18O_sw and d17O_sw
        initial_guess = [0.0, 0.0]
        
        # Solve the equations
        solution = fsolve(equations_to_solve, initial_guess, args=(fF_sp, fF_sfw, fF_cw, fF_cg, fF_r, fF_c, fF_qz, cT, qzT))
        
        # Append results to list
        results.append({
            'F_sp': fF_sp,
            'F_sfw': fF_sfw,
            'F_cw': fF_cw,
            'F_cg': fF_cg,
            'F_r': fF_r,
            'F_c': fF_c,
            'F_qz': fF_qz,
            'cT': cT,
            'qzT': qzT,
            'd18Osw': solution[0],
            'd17Osw': solution[1]
        })
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    return df

# Calculate modern steady state
initial_guess = [0.0, 0.0]
mss = fsolve(equations_to_solve, initial_guess, args=(1, 1, 1, 1, 1, 1, 1, 150, 150))
print(f"Modern steady state: d18O = {mss[0]:.2f}, Dp17O = {Dp17O(mss[1], mss[0]):.1f}")


# Run Monte Carlo simulation
simulation_results = monte_carlo_simulation(200000)
simulation_results['Dp17Osw'] = Dp17O(simulation_results['d17Osw'], simulation_results['d18Osw'])
simulation_results.to_csv(os.path.join(sys.path[0], 'DP Table S3.csv'), index=False)

print("Monte-carlo sewater modelling complete!\n")


############################################################################################################
# Define the functions nedded to work with the carbonate data

def a18c(T):

    # Vasconcelos et al. (2016) - dolomite
    return np.exp((2.73 * 10**6 / T**2 + 0.26) / 1000)

    # Hayles et al. (2018) - dolomite
    # B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_dolomite) / np.exp(B_water)


def theta_c(T):

    # Hayles et al. (2018) - dolomite
    K_dolomite = 9.937692E+08 / T**5 + -2.069620E+07 / T**4 + 1.653613E+05 / T**3 + -5.704833E+02 / T**2 + 1.462601E-01 / T + 0.5304874
    B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04
    K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    a18 = np.exp(B_dolomite) / np.exp(B_water)
    return K_dolomite + (K_dolomite-K_water) * (B_water / np.log(a18))


# Read calcite data from CSV file
carbonates = pd.read_csv(os.path.join(sys.path[0], "DP Table S2.csv"))
carbonates = carbonates.rename(columns={"d18O_AC": "d18O", "d17O_AC": "d17O", "Dp17O_AC": "Dp17O"})

# Filter data
carbonates = carbonates[carbonates["Mineralogy"] == "dolomite"]

# Read in possible seawater compositions from CSV file
all_fluids = simulation_results
all_fluids = all_fluids[all_fluids["Dp17Osw"] <= 20]

# Initialize lists to store calculated values
sum_distances = []
avg_temperatures = []
min_temperatures = []
max_temperatures = []

# Iterate over the modeled fluids to calculate how well they fit the dolomite data
for _, row in tqdm(all_fluids.iterrows(), total=len(all_fluids)):
    d18Osw = row["d18Osw"]
    d17Osw = row["d17Osw"]
    Dp17Ow = row["Dp17Osw"]

    # Calculate equilibrium points between 0 °C and 300 °C with 1 degree resolution
    equilibrium_temperatures = np.arange(0, 300, 0.1) + 273.15
    d18O_mineral = d18Oc(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Oc(equilibrium_temperatures, d17Osw)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T

    data = []
    for i, row in carbonates.iterrows():
        A = np.array([row["d18O"], row["Dp17O"]])
        distances = np.linalg.norm(mineral_equilibrium[:, :2] - A, axis=1)
        mindist = np.min(distances)
        closest_index = np.argmin(distances)
        closest_point = mineral_equilibrium[closest_index]
        tempera = closest_point[2]
        data.append({"distances": mindist, "temperatures": tempera})

    # Calculate aggregate values for this row
    sum_distance = np.sum([entry["distances"] for entry in data])
    avg_temperature = np.mean([entry["temperatures"] - 273.15 for entry in data]).round(2)
    min_temperature = np.min([entry["temperatures"] - 273.15 for entry in data]).round(2)
    max_temperature = np.max([entry["temperatures"] - 273.15 for entry in data]).round(2)

    # Append calculated values to the respective lists
    sum_distances.append(sum_distance)
    avg_temperatures.append(avg_temperature)
    min_temperatures.append(min_temperature)
    max_temperatures.append(max_temperature)

# Add new columns to the existing DataFrame
all_fluids["sum_distance"] = sum_distances
all_fluids["avg_temperature"] = avg_temperatures
all_fluids["min_temperature"] = min_temperatures
all_fluids["max_temperature"] = max_temperatures

# Define the cut-off values
sum_distance_cutoff = all_fluids['sum_distance'].quantile(0.01)
print(f"Cut-off value for sum_distance: {sum_distance_cutoff:.3f}")
all_fluids['fits'] = np.where(all_fluids["sum_distance"] <= sum_distance_cutoff, 'y', 'n')
all_fluids.to_csv(os.path.join(sys.path[0], 'DP Table S4.csv'), index=False)

print("Obtaining best-fit compositions complete!")
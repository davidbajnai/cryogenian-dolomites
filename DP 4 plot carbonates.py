# This code is used to:
# Plot the data in triple oxygen isotope space

# INPUT:  DP Table S2.csv (carbonate data)
#         DP Table S4.csv (best-fit compositions)

# OUTPUT: DP Figure 2a.png

# >>>>>>>>>

# Import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from curlyBrace import curlyBrace
import subprocess
from functions import *

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Define additional functions
def a18cal(T):

    # Daeron et al. (2019) - calcite
    return np.exp((17.57 * 1000 / T - 29.13) / 1000)

    # Hayles et al. (2018) - calcite
    # B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / T**4 + -4.495755E+05 / T**3  + 1.307870E+04 / T**2 + -5.393675E-01 / T + 1.331245E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_calcite) / np.exp(B_water)


def a18dol(T):

    # Vasconcelos et al. (2016) - dolomite
    return np.exp((2.73 * 10**6 / T**2 + 0.26) / 1000)

    # Hayles et al. (2018) - dolomite
    # B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_dolomite) / np.exp(B_water)


def theta_cal(T):

    # Wostbrock et al. (2020) - calcite
    # return -1.39 / T + 0.5305

    # Hayles et al. (2018) - calcite
    K_calcite = 1.019124E+09 / T**5 + -2.117501E+07 / T**4 + 1.686453E+05 / T**3 + -5.784679E+02 / T**2 + 1.489666E-01 / T + 0.5304852
    B_calcite = 7.027321E+14 / T**7 + -1.633009E+13 / T**6 + 1.463936E+11 / T**5 + -5.417531E+08 / T**4 + -4.495755E+05 / T**3  + 1.307870E+04 / T**2 + -5.393675E-01 / T + 1.331245E-04
    K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    a18 = np.exp(B_calcite) / np.exp(B_water)
    return K_calcite + (K_calcite-K_water) * (B_water / np.log(a18))


def theta_dol(T):

    # Hayles et al. (2018) - dolomite
    K_dolomite = 9.937692E+08 / T**5 + -2.069620E+07 / T**4 + 1.653613E+05 / T**3 + -5.704833E+02 / T**2 + 1.462601E-01 / T + 0.5304874
    B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04

    K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03

    a18 = np.exp(B_dolomite) / np.exp(B_water)
    return K_dolomite + (K_dolomite-K_water) * (B_water / np.log(a18))


def a17cal(T):
    return a18cal(T) ** theta_cal(T)


def d18Ocal(equilibrium_temperatures, d18Osw):
    return a18cal(equilibrium_temperatures) * (d18Osw + 1000) - 1000


def d17Ocal(equilibrium_temperatures, d17Ow):
    return a17cal(equilibrium_temperatures) * (d17Ow + 1000) - 1000


def a17dol(T):
    return a18dol(T) ** theta_dol(T)


def d18Odol(equilibrium_temperatures, d18Osw):
    return a18dol(equilibrium_temperatures) * (d18Osw + 1000) - 1000


def d17Odol(equilibrium_temperatures, d17Ow):
    return a17dol(equilibrium_temperatures) * (d17Ow + 1000) - 1000


def plot_dolomite_equilibrium(Dp17Osw, d18Osw, Tmin, Tmax, ax, color="k"):
    d17Ow = unprime(0.528 * prime(d18Osw) + Dp17Osw / 1000)

    ax.scatter(prime(d18Osw), Dp17O(d17Ow, d18Osw),
               marker="*", fc="w", ec="k", zorder=10, s=100)

    # equilibrium, highlight range
    equilibrium_temperatures = np.arange(Tmin, Tmax, 0.5) + 273.15
    d18O_mineral = d18Odol(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Odol(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.plot(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
            c=color, ls = "-", lw = .8, zorder=0)

    # equilibrium, highlight range, marker every 10 °C
    equilibrium_temperatures = np.arange(Tmin, Tmax + 1, 10) + 273.15
    d18O_mineral = d18Odol(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Odol(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
               s=15, marker=".", fc="white", ec=color, zorder=0)

    # Return equilibrium data as a dataframe every 5 °C
    equilibrium_temperatures = np.arange(Tmin, Tmax + 1, 5) + 273.15
    d18O_mineral = d18Odol(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Odol(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    equilibrium_df = pd.DataFrame(mineral_equilibrium)
    equilibrium_df[2] = equilibrium_df[2] - 273.15
    equilibrium_df = equilibrium_df.rename(columns={0: "d18O", 1: "Dp17O", 2: "temperature"})
    return equilibrium_df


def plot_calcite_equilibrium(Dp17Osw, d18Osw, Tmin, Tmax, ax, color="k"):
    d17Ow = unprime(0.528 * prime(d18Osw) + Dp17Osw / 1000)

    # equilibrium, highlight range
    equilibrium_temperatures = np.arange(Tmin, Tmax, 0.5) + 273.15
    d18O_mineral = d18Ocal(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Ocal(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.plot(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
            c=color, ls=":", zorder=0)

    # equilibrium, highlight range, marker every 10 °C
    equilibrium_temperatures = np.arange(Tmin, Tmax + 1, 10) + 273.15
    d18O_mineral = d18Ocal(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17Ocal(equilibrium_temperatures, d17Ow)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
    ax.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
               s=15, marker=".", fc="white", ec=color, zorder=0)

    # Return equilibrium data as a dataframe
    equilibrium_df = pd.DataFrame(mineral_equilibrium)
    equilibrium_df[2] = equilibrium_df[2] - 273.15
    equilibrium_df = equilibrium_df.rename(columns={0: "d18O", 1: "Dp17O", 2: "temperature"})
    return equilibrium_df


# Read in data
data = pd.read_csv(os.path.join(sys.path[0], "DP Table S2.csv"))
data["Dp17O_error"] = 9
modelwater = pd.read_csv(os.path.join(sys.path[0], "DP Table S4.csv"))
modelwater = modelwater[modelwater["fits"] == "y"]

# Start plotting
fig, ax1 = plt.subplots(1, 1)

# Plot modeled fluids color coded by temperature
ax1.scatter(prime(modelwater["d18Osw"]), modelwater["Dp17Osw"],
            marker=".", c="#1455C0", zorder=0)


# Plot dolomite equilibrium - using median d18Osw
modelwater = modelwater.sort_values(by="d18Osw").reset_index()
index = len(modelwater) // 2
d18O_ocean = modelwater.loc[index, "d18Osw"]
Dp17O_ocean = modelwater.loc[index, "Dp17Osw"]
print(f"The median is  {d18O_ocean:.0f}‰ and {Dp17O_ocean:.0f} ppm")

df_eq_dol = plot_dolomite_equilibrium(
    Dp17O_ocean, d18O_ocean, 0, 80, ax1, color="k")

ax1.annotate(f"{df_eq_dol.iloc[0, 2]:.0f} °C",
    (prime(df_eq_dol.iloc[0, 0]), df_eq_dol.iloc[0, 1]),
    xytext=(prime(df_eq_dol.iloc[0, 0]) + 4, df_eq_dol.iloc[0, 1]),
    ha="left", va="center",
    bbox=dict(fc="white", ec="none", pad=0.2),
    arrowprops=dict(arrowstyle="-|>", color="k"))
ax1.annotate(f"{df_eq_dol.iloc[-1, 2]:.0f} °C",
    (prime(df_eq_dol.iloc[-1, 0]), df_eq_dol.iloc[-1, 1]),
    xytext=(prime(df_eq_dol.iloc[-1, 0]) + 4, df_eq_dol.iloc[-1, 1] ),
    ha="left", va="center",
    bbox=dict(fc="white", ec="none", pad=0.2),
    arrowprops=dict(arrowstyle="-|>", color="k"))

curlyBrace(fig, ax1, [df_eq_dol.iloc[-1,0], df_eq_dol.iloc[-1,1]+4], [df_eq_dol.iloc[0,0], df_eq_dol.iloc[-1,1]+4],
           0.05, str_text="dolomite equilibrium", int_line_num=2, color='k')

curlyBrace(fig, ax1, [max(modelwater['d18Osw'])+2, max(modelwater['Dp17Osw'])], [max(modelwater['d18Osw'])+2, min(modelwater['Dp17Osw'])],
           0.1,  str_text="", int_line_num=4, color='k')

# ax1.text at the median
ax1.text(max(modelwater['d18Osw'])+5, (max(modelwater['Dp17Osw'])+min(modelwater['Dp17Osw']))/2,
         "seawater\n(see also in " + r"$\bf{b}$)", va="center", ha="left", color="k")


# Meteoric alteration
d18O_unaltered = data["d18O_AC"].max()
Dp17O_unaltered = data[data["d18O_AC"] == d18O_unaltered]["Dp17O_AC"].iloc[0]

# Composition of meteoric water
d18O_meteoric = -9-6 # modern average minus -6‰ shift in the ocean
Dp17O_meteoric = 25
diafluid_d17O = unprime(Dp17O_meteoric / 1000 + 0.528 * prime(d18O_meteoric))
temp_range = np.array([200])

for dia_temp in temp_range:
    dia_em_d18O = d18Ocal(dia_temp + 273.15, d18O_meteoric)
    dia_em_d17O = d17Ocal(dia_temp + 273.15, diafluid_d17O)
    dia_em_Dp17O = Dp17O(dia_em_d17O, dia_em_d18O)

    mix = mix_d17O(d18O_A=d18O_unaltered, D17O_A=Dp17O_unaltered,
                d18O_B=dia_em_d18O, D17O_B=dia_em_Dp17O)
    ax1.plot(prime(mix["mix_d18O"][:-30]), mix["mix_Dp17O"][:-30],
            c="k", zorder=0, ls = ":",lw=.8)
    ax1.annotate("",
                xy=(prime(mix["mix_d18O"].iloc[-32]), mix["mix_Dp17O"].iloc[-32]),
                xytext=(prime(mix["mix_d18O"].iloc[-30]),mix["mix_Dp17O"].iloc[-30]),
                ha="center", va="center", zorder = 2, color = "k",
                arrowprops=dict(arrowstyle="<|-", color="k", lw=.8))
ax1.text(prime(mix["mix_d18O"].iloc[60]), mix["mix_Dp17O"].iloc[60],
            f"post-depositional\nalteration",
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8),
            va="center", ha="center", color="k")

# # add star to the 50% point
positions = [10, 20, 30, 40, 50]
for pos in positions:
    ax1.scatter(prime(mix["mix_d18O"].iloc[pos]), mix["mix_Dp17O"].iloc[pos],
                marker=".", fc="k", ec="k", zorder=100)

positions = [10, 50]
for pos in positions:
    ax1.text(prime(mix["mix_d18O"].iloc[pos]), mix["mix_Dp17O"].iloc[pos]-2,
                f"{mix['xB'].iloc[pos]:.0f}%", va="top", ha="center", color="k",
                bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))



# Plot carbonate samples
colName = "Mineralogy"
data["Mineralogy"] = data["Mineralogy"].replace("calcite", "limestone")
cat_col = (data[colName].unique())
colors = dict(zip(cat_col, plt.cm.Wistia(
    np.linspace(0, 1, len(cat_col)))))
markName = "Mineralogy"
cat_mark = data[markName].unique()
markers = dict(zip(cat_mark, ["o", "s", "s", "D", "^", "v", "P", "X"]))

for col in cat_col:
    for mark in cat_mark:
        dat = data[(data[colName] == col) & (data[markName] == mark)]
        if len(dat) > 0:
            ax1.scatter(prime(dat["d18O_AC"]), dat["Dp17O_AC"],
                        marker=markers[mark], color=colors[col], ec="k", zorder=10)
            ax1.errorbar(prime(dat["d18O_AC"]), dat["Dp17O_AC"],
                         xerr=dat["d18O_error"], yerr=dat["Dp17O_error"],
                         fmt="none", color=colors[col])
            # Indicate sample names
            # for i, txt in enumerate(dat["SampleName"]):
            #         ax1.annotate(txt, (prime(dat["d18O_AC"].iloc[i]), dat["Dp17O_AC"].iloc[i]),
            #                     xytext=(prime(dat["d18O_AC"].iloc[i]) + 0, dat["Dp17O_AC"].iloc[i] + 0),
            #                     ha="center", va="center", color="k", fontsize=2, zorder=11)

# Mark samples AQ24 and Z_MC with an asterisk
for i, txt in enumerate(data["SampleName"]):
    if txt == "AQ24" or txt == "Z_MC":
        ax1.scatter(prime(data["d18O_AC"].iloc[i]), data["Dp17O_AC"].iloc[i],
                    marker="$*$", fc="w", ec="none", zorder=10)

# Plot error bar
d18O_error = data["d18O_error"].mean()
Dp17O_error = data["Dp17O_error"].mean()
print(f"The average measurement errors are {d18O_error:.2f}‰ for d18O and {Dp17O_error:.0f} ppm for Dp17O")

for mark, col in zip(cat_mark, cat_col):
        ax1.scatter([], [], marker=markers[mark], facecolor=colors[col], edgecolor="black",label=col)
ax1.legend(loc='lower left')

# Axis properties
ax1.set_ylim(-105, 5)
ax1.set_xlim(-15, 45)
ax1.set_ylabel("$\Delta\prime^{17}$O (ppm)")
ax1.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW)")
ax1.text(0.02, 0.98, "a", size=14, ha="left", va="top",
         transform=ax1.transAxes, fontweight="bold")

plt.savefig(os.path.join(sys.path[0], "DP Figure 2a"))
plt.close("all")

subprocess.run([sys.executable, os.path.join(sys.path[0], "combineImage.py")])
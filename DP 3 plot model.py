# This code is used to create:
# Figure 2b, Figure 3, Figure 4, Figure S2, Figure S4

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from curlyBrace import curlyBrace
from tqdm import tqdm

# Import functions
from functions import *

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["savefig.bbox"] = "tight"

# Define additional functions
def a18_carb(T):

    # Daeron et al. (2019) - calcite
    # return np.exp((17.57 * 1000 / T - 29.13) / 1000)

    # Vasconcelos et al. (2016) - dolomite
    return np.exp((2.73 * 10**6 / T**2 + 0.26) / 1000)

    # Hayles et al. (2018) - dolomite
    # B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04
    # B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03
    # return np.exp(B_dolomite) / np.exp(B_water)


def theta_carb(T):

    # Wostbrock et al. (2020) - calcite
    # return -1.39 / T + 0.5305

    # Hayles et al. (2018) - dolomite
    K_dolomite = 9.937692E+08 / T**5 + -2.069620E+07 / T**4 + 1.653613E+05 / T**3 + -5.704833E+02 / T**2 + 1.462601E-01 / T + 0.5304874
    B_dolomite = 6.981231E+14 / T**7 + -1.625341E+13 / T**6 + 1.461088E+11 / T**5 + -5.437285E+08 / T**4 + -4.352597E+05 / T**3  + 1.320284E+04 / T**2 + -5.279219E-01 / T + 1.304577E-04

    K_water = 7.625734E+06 / T**5 + 1.216102E+06 / T**4 + -2.135774E+04 / T**3 + 1.323782E+02 / T**2 + -4.931630E-01 / T + 0.5306551
    B_water = -6.705843E+15 / T**7 + 1.333519E+14 / T**6 + -1.114055E+12 / T**5 + 5.090782E+09 / T**4 + -1.353889E+07 / T**3 + 2.143196E+04 / T**2 + 5.689300 / T + -7.839005E-03

    a18 = np.exp(B_dolomite) / np.exp(B_water)
    return K_dolomite + (K_dolomite-K_water) * (B_water / np.log(a18))


def a17_carb(T):
    return a18_carb(T)**theta_carb(T)


def d18O_carb(equilibrium_temperatures, d18Osw):
    return a18_carb(equilibrium_temperatures) * (d18Osw+1000) - 1000


def d17O_carb(equilibrium_temperatures, d18Osw):
    return a17_carb(equilibrium_temperatures) * (d18Osw+1000) - 1000


# Read data from the files
dfsw = pd.read_csv(sys.path[0] + "/seawater.csv", delimiter=',')
dfcarb = pd.read_csv(sys.path[0] + "/DP Table S2.csv", delimiter=',')
dfcarb = dfcarb[dfcarb["Mineralogy"] == "dolomite"]
dfA = pd.read_csv(sys.path[0] + "/DP Table S3.csv", delimiter=',')
dfC = pd.read_csv(sys.path[0] + "/DP Table S4.csv", delimiter=',')
dfB = dfC[dfC['fits'] == "y"]

# Print out the average composition of modern seawater
print(f"Modern seawater average values: {dfsw['d18O'].mean():.2f}‰ and {dfsw['Dp17O'].mean():.0f} ppm")

############################ Figure 4 ############################
all_temperatures = []
for _, row in tqdm(dfB.iterrows(), total=len(dfB)):
    d18Osw = row["d18Osw"]
    d17Osw = row["d17Osw"]
    Dp17Ow = row["Dp17Osw"]

    equilibrium_temperatures = np.arange(0, 300, 0.1) + 273.15
    d18O_mineral = d18O_carb(equilibrium_temperatures, d18Osw)
    d17O_mineral = d17O_carb(equilibrium_temperatures, d17Osw)
    mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T

    for i, row in dfcarb.iterrows():
        A = np.array([row["d18O_AC"], row["Dp17O_AC"]])
        distances = np.linalg.norm(mineral_equilibrium[:, :2] - A, axis=1)
        mindist = np.min(distances)
        closest_index = np.argmin(distances)
        closest_point = mineral_equilibrium[closest_index]
        all_temperatures.append({"SampleName": row["SampleName"], "temperatures": closest_point[2]-273.15})


fig = plt.figure(figsize=(3, 3.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.01)

# Top subplot
ax0 = fig.add_subplot(gs[0])
ax0.axis('off')
ax0.set_xlim(5, 85)
ax0.set_ylim(0.2, 1.2)

algae_range = [55, 60]
plt.plot(algae_range, [0.5, 0.5], color='#63A615', lw=10, solid_capstyle='butt')
plt.text(np.mean(algae_range), 0.5, '✝', ha='center', va='center', c = "w")
algae_range = [20, 30]
plt.plot(algae_range, [0.5, 0.5], color='#63A615', lw=10, solid_capstyle='butt')
plt.text(np.mean(algae_range), 0.5, '✔', ha='center', va='center', c = "w")
plt.text(np.max(algae_range)+1, 0.5, 'algae', ha='left', va='center', c = "#63A615")

cyanobacteria_range = [70, 73]
plt.plot(cyanobacteria_range, [1,1], color='#309FD1', lw=10, solid_capstyle='butt')
plt.text(np.mean(cyanobacteria_range),1, '✝', ha='center', va='center', c = "w")
cyanobacteria_range = [25, 30]
plt.plot(cyanobacteria_range, [1,1], color='#309FD1', lw=10, solid_capstyle='butt')
plt.text(np.mean(cyanobacteria_range), 1, '✔', ha='center', va='center', c = "w")
plt.text(np.max(cyanobacteria_range)+1, 1, 'cyanobacteria', ha='left', va='center', c = "#309FD1")


# Bottom subplot
ax1 = fig.add_subplot(gs[1], sharex=ax0)
all_temps = np.array([entry["temperatures"] for entry in all_temperatures])
filtered_temps = np.array([entry["temperatures"] for entry in all_temperatures if entry["SampleName"] in ["AQ24", "Z_MC"]])

bins = np.arange(0, 70, 5)
n1, _, _ = ax1.hist(all_temps, bins=bins, color='#858379',
                    ec='w', label='all dolomites')
n2, _, _ = ax1.hist(filtered_temps, bins=bins, color='#1455C0',
                    ec='w', label=r'highest-$\delta^{18}$O dolomites')

if np.max(all_temps) > np.max(bins) or np.min(all_temps) < np.min(bins):
    print("The histogram bin range is too narrow. Increase the bin size!")
    sys.exit()

ax1.text(32.5, 1500, "all dolomites",
         ha='left', va='center', color='k',
         bbox=dict(fc='w', pad=1.2, lw=1, alpha=0.5, ec='k'))

ax1.text(22.5, 300, "highest-$\delta^{18}$O\ndolomites",
         ha='left', va='center', color='k',
         bbox=dict(fc='w', pad=1.2, lw=1, alpha=0.5, ec='#1455C0'))

ax1.set_xlabel('Calculated seawater temperatures (°C)')
ax1.set_ylabel('# of estimates')
ax1.yaxis.set_ticks([])
ax1.set_xticks(np.arange(10, 90, 10))

plt.savefig(os.path.join(sys.path[0], "DP Figure 4"))
plt.close()



############################ Figure S2 ############################
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

# Subplot a
dfB = dfB.sort_values(by="d18Osw").reset_index()
index = len(dfB) // 2
d18O_median = dfB.loc[index, "d18Osw"]
Dp17O_median = dfB.loc[index, "Dp17Osw"]
d17O_median = d17O(d18O_median, Dp17O_median)

print(f"Among the best-fit seawater compositions, the two end-members have δ18Osw and ∆17Osw values of {min(dfB['d18Osw']):.0f}‰ and {max(dfB['Dp17Osw']):.0f} ppm and {max(dfB['d18Osw']):.0f}‰ and {min(dfB['Dp17Osw']):.0f} ppm,\nrespectively, with median values of {d18O_median:.0f}‰ and {Dp17O_median:.0f} ppm")

ax1.scatter(prime(d18O_median), Dp17O_median,
            marker="o", c="#347DE0",
            label="a modelled seawater")
ax1.scatter(prime(dfcarb['d18O_AC']), dfcarb['Dp17O_AC'],
            marker="s", fc="#F39200", ec="k",
            label="dolomites")


# Calculate equilibrium points between 0 °C and 300 °C with 1 degree resolution
equilibrium_temperatures = np.arange(0, 300, 0.1) + 273.15
d18O_mineral = d18O_carb(equilibrium_temperatures, d18O_median)
d17O_mineral = d17O_carb(equilibrium_temperatures, d17O_median)
mineral_equilibrium = np.array([d18O_mineral, Dp17O(d17O_mineral, d18O_mineral), equilibrium_temperatures]).T
ax1.scatter(prime(mineral_equilibrium[:, 0]), mineral_equilibrium[:, 1],
         c="k", marker = ".", s = 5, label="equilibrium points")

all_temperatures = []
for i, row in dfcarb.iterrows():
    A = np.array([row["d18O_AC"], row["Dp17O_AC"]])
    distances = np.linalg.norm(mineral_equilibrium[:, :2] - A, axis=1)
    mindist = np.min(distances)
    closest_index = np.argmin(distances)
    closest_point = mineral_equilibrium[closest_index]

    ax1.plot([prime(A[0]), prime(closest_point[0])],
             [A[1], closest_point[1]],
             c='#DB0078')
    ax1.text(prime(closest_point[0])+0.3, closest_point[1],
             f"{(closest_point[2]-273.15):.0f} °C",
             ha="left", va="center", fontsize=3,
             bbox=dict(facecolor='w', pad=0.2, lw=0, alpha=0.5))

ax1.plot(0,0, c = '#DB0078', label = "Euclidean distance")
ax1.legend(loc="lower left")

ax1.set_xlabel("$\delta^{\prime 18}$O (‰, VSMOW)")
ax1.set_ylabel("$\Delta^{\prime 17}$O (ppm)")
ax1.set_xlim(-6, 36)
ax1.text(0.98, 0.98, "a", size=14, ha="right", va="top", transform=ax1.transAxes, fontweight="bold")

# Subplot b
ax2.scatter(dfC['sum_distance'], dfC['Dp17Osw'],
            marker=".", c="#BCBBB2",
            label="modelled seawaters")
ax2.scatter(dfB['sum_distance'], dfB['Dp17Osw'],
            marker=".", c="#1455C0",
            label="best-fit to dolomites")

# Cut-off lines
cutoff = 36.168
ax2.axvline(x=cutoff, color='k', linestyle='--', lw=0.5)
ax2.axhline(y=20, color='k', linestyle='--', lw=0.5)

ax2.legend(loc="lower right")

ax2.set_xlabel('Cumulative distances')
ax2.set_ylabel("$\Delta^{\prime 17}$O (ppm)")
ax2.text(0.98, 0.98, "b", size=14, ha="right", va="top", transform=ax2.transAxes, fontweight="bold", bbox=dict(facecolor='w', pad=0.2, lw = 0, alpha=0.5))

plt.savefig(os.path.join(sys.path[0], "DP Figure S2"))
plt.close()


############################ Figure 2b ############################
fig, ax = plt.subplots(1, 1)
ax.scatter(prime(dfA['d18Osw']), dfA['Dp17Osw'],
           marker=".", c="#BCBBB2", alpha=0.5,
           label="model")
ax.scatter(prime(dfsw['d18O']), dfsw['Dp17O'],
              marker=".", c="#282D37", alpha=0.5,
              label="modern")
ax.scatter(prime(dfB['d18Osw']), dfB['Dp17Osw'],
              marker=".", c="#1455C0", alpha=0.5,
              label="best-fit")

ax.scatter(prime(-0.24), -17.4,
           marker="*", fc="k", ec="w", s=100,
           label="steady state")
ax.scatter(prime(d18O_median), Dp17O_median,
           marker="*", fc="w", ec="k", s=100,
           label="median best-fit")

curlyBrace(fig, ax, [max(dfB['d18Osw']), min(dfB['Dp17Osw'])-1], [min(dfB['d18Osw']), min(dfB['Dp17Osw'])-1],
           0.1, str_text="best-fit\ncompositions", int_line_num=2.5, color='k', lw=1)

curlyBrace(fig, ax, [min(dfA['d18Osw']), max(dfA['Dp17Osw'])+1], [max(dfA['d18Osw']), max(dfA['Dp17Osw'])+1],
           0.05,  str_text="modelled seawater", int_line_num=2, color='k', lw=1)

curlyBrace(fig, ax, [min(dfsw['d18O']), max(dfsw['Dp17O'])+2], [max(dfsw['d18O']), max(dfsw['Dp17O'])+2],
           0.1,  str_text="modern\nseawater", int_line_num=3, color='k', lw=1)


ax.set_xlim(-9.5, 5.5)
ax.set_ylim(-35, 75)

ax.set_xlabel("$\delta^{\prime 18}$O (‰, VSMOW)")
ax.set_ylabel("$\Delta^{\prime 17}$O (ppm)")

ax.text(0.98, 0.98, "b", size=14, ha="right", va="top",
        transform=ax.transAxes, fontweight="bold")

plt.savefig(os.path.join(sys.path[0], "DP Figure 2b"))
print('Figure "DP Figure 2b" saved')
plt.close()



def plot_fluxes(df, filename):
    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9.5, 9.5), constrained_layout=True)

    sc = ax1.scatter(prime(df['d18Osw']), df['Dp17Osw'],
                marker=".", c=df['F_sp'], cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('$F_{sp}$ (18.3x10$^{15}$ g yr$^{-1}$)')
    cbar.ax.yaxis.set_label_position('right')
    ax1.text(0.98, 0.98, "a", size=14, ha="right", va="top",
            transform=ax1.transAxes, fontweight="bold")
    
    sc = ax2.scatter(prime(df['d18Osw']), df['Dp17Osw'],
                marker=".", c=df['F_cw'], cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('$F_{cw}$ (10x10$^{15}$ g yr$^{-1}$)')
    cbar.ax.yaxis.set_label_position('right')
    ax2.text(0.98, 0.98, "b", size=14, ha="right", va="top",
            transform=ax2.transAxes, fontweight="bold")

    sc = ax3.scatter(prime(df['d18Osw']), df['Dp17Osw'],
                marker=".", c=df['F_sfw'], cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('$F_{sfw}$ (2.2x10$^{15}$ g yr$^{-1}$)')
    cbar.ax.yaxis.set_label_position('right')
    ax3.text(0.98, 0.98, "c", size=14, ha="right", va="top",
            transform=ax3.transAxes, fontweight="bold")

    sc = ax4.scatter(prime(df['d18Osw']), df['Dp17Osw'],
                marker=".", c=df['F_c'], cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('$F_{carb}$ and $F_{SiO_2}$ (0.077x10$^{15}$ g yr$^{-1}$)')
    cbar.ax.yaxis.set_label_position('right')
    ax4.text(0.98, 0.98, "d", size=14, ha="right", va="top",
            transform=ax4.transAxes, fontweight="bold")


    axes_list = [ax1, ax2, ax3, ax4]
    for ax in axes_list:
        ax.set_xlabel("$\delta^{\prime 18}$O (‰, VSMOW)")
        ax.set_ylabel("$\Delta^{\prime 17}$O (ppm)")


    # Calcualte median factors
    filtered_columns = [col for col in df.columns if "F_" in col and "F_qz" not in col]
    results = df.loc[:, filtered_columns].median()
    factors = results.tolist()
    fluxes = [18.3, 2.2, 10, 1.5, 0.8, 0.0768]
    fluxes_names = ['F$_{sp}$', 'F$_{sfw}$', 'F$_{cw}$', 'F$_{cg}$', 'F$_r$', "F$_{carb}$ & F$_{SiO_2}$"]
    fluxes_times_factors = [flux * factor for flux, factor in zip(fluxes, factors)]

    # Set log scale for both axes
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    # Add a 1:1 line
    x = np.logspace(-2, 2, 100)
    ax5.plot(x, x, c="#9C9A8E", ls = '--', zorder = -1)

    # Plot the points
    ax5.scatter(fluxes, fluxes_times_factors,
                marker="o", fc='#1455C0', ec = "k")

    # Label each point
    for i, name in enumerate(fluxes_names):
        ax5.text(fluxes[i], fluxes_times_factors[i]*1.5, name, ha='center', va = "bottom", fontstyle='italic')

    ax5.annotate("Neoproterozoic fluxes\nlower than modern",
                 xy=(0.35, 0.35), xycoords='axes fraction', xytext=(0.46, 0.2), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="<|-", lw = 3, color = "k"), ha='center', va='center')
    ax5.annotate("Neoproterozoic fluxes\nhigher than modern",
                 xy=(0.35, 0.35), xycoords='axes fraction', xytext=(0.24, 0.5), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="<|-", lw = 3, color = "k"), ha='center', va='center')

    # Set labels and title
    ax5.set_xlabel('Modern fluxes (10$^{15}$ g yr$^{-1}$)')
    ax5.set_ylabel('Median of modelled fluxes (10$^{15}$ g yr$^{-1}$)')
    ax5.text(0.98, 0.98, "e", size=14, ha="right", va="top",
            transform=ax5.transAxes, fontweight="bold")


    # make ax6 empty
    ax6.axis('off')

    plt.savefig(os.path.join(sys.path[0], filename))
    print('Figure "' + filename +'" saved')
    plt.close()

plot_fluxes(dfB, 'DP Figure 3')
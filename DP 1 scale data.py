# This code is used to:
# Scale the TILDAS data based on the accepted values of the in-house reference gases

# INPUT:    DP Table S2.csv (replicate data)
#           DP Table 2.csv (sample info)

# OUTPUT:   DP Figure S1.png
#           DP Table S3.csv (scaled and averaged data)

# >>>>>>>>>

# Import libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from functions import *

# Plot parameters
plt.rcParams.update({'font.size': 7})
plt.rcParams['scatter.edgecolors'] = "k"
plt.rcParams['scatter.marker'] = "o"
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Define additional functions

# Function to print info for scaled samples
def print_info(df, d18O_col, Dp17O_col, sample_name):
    gas_subset = df[df["SampleName"].str.contains(sample_name)].copy()
    d18O_mean = gas_subset[d18O_col].mean()
    d18O_std = gas_subset[d18O_col].std()
    Dp17O_mean = gas_subset[Dp17O_col].mean()
    Dp17O_std = gas_subset[Dp17O_col].std()
    N_gas = len(gas_subset)
    print(f"{sample_name}, N = {N_gas}, d18O = {d18O_mean:.3f}(±{d18O_std:.3f})‰, ∆'17O = {Dp17O_mean:.0f}(±{Dp17O_std:.0f}) ppm", end="")


# Function to apply the acid fractionation factor based on the mineralogy
def applyAFF(d18O_CO2, d17O_CO2, mineral):

    # Acid fractionation correction
    if mineral == "calcite":
        a18 = 1.01025

    elif mineral == "aragonite":
        a18 = 1.01063
    
    elif mineral == "dolomite":
        # a18 = np.exp(11.03/1000) # Sharma and Clayton (1965)
        a18 = np.mean([1.01178, 1.01186]) # Rosenbaum and Sheppard (1986)

    d18O_AC = (d18O_CO2 + 1000) / a18 - 1000
    d17O_AC = (d17O_CO2 + 1000) / (a18 ** 0.523) - 1000
    Dp17O_AC = Dp17O(d17O_AC, d18O_AC)

    return d18O_AC, d17O_AC, Dp17O_AC


# This function scales the data based on the accepted values of the light and heavy reference gases
# The scaling is done for each measurement period separately
def scaleData(df, project):

    df["dateTimeMeasured"] = pd.to_datetime(df["dateTimeMeasured"])

    # Perform the scaling for each meausrement period separately
    df_samples = pd.DataFrame()
    grouped = df.groupby("measurementPeriod")
    if grouped.ngroups == 1:
        SuppFig = [""]
    else:
        SuppFig = ["a","b","c","d"]
    FigNum = 0
    for period, group in grouped:

        print(f"\nMeasurement period {period}:")
        print_info(group, "d18O", "Dp17O", "light"); print("\t<--- unscaled")
        print_info(group, "d18O", "Dp17O", "heavy"); print("\t<--- unscaled")

        # Do the scaling here, based on the accepted values of the light and heavy reference gases

        # Measured CO2 values
        heavy_d18O_measured = group[group["SampleName"].str.contains("heavy")]["d18O"].mean()
        heavy_d17O_measured = group[group["SampleName"].str.contains("heavy")]["d17O"].mean()

        light_d18O_measured = group[group["SampleName"].str.contains("light")]["d18O"].mean()
        light_d17O_measured = group[group["SampleName"].str.contains("light")]["d17O"].mean()

        # Accepted CO2 values - see Bajnai et al. (2024, Chem Geol) for details
        heavy_d18O_accepted = 76.820
        heavy_Dp17O_accepted = -213
        heavy_d17O_accepted = d17O(heavy_d18O_accepted, heavy_Dp17O_accepted)

        light_d18O_accepted = -1.509
        light_Dp17O_accepted = -141
        light_d17O_accepted = d17O(light_d18O_accepted, light_Dp17O_accepted)

        # Calculate the scaling factors
        slope_d18O = (light_d18O_accepted - heavy_d18O_accepted) / (light_d18O_measured - heavy_d18O_measured)
        intercept_d18O = heavy_d18O_accepted - slope_d18O * heavy_d18O_measured

        slope_d17O = (light_d17O_accepted - heavy_d17O_accepted) / (light_d17O_measured - heavy_d17O_measured)
        intercept_d17O = heavy_d17O_accepted - slope_d17O * heavy_d17O_measured

        # Scale the measured values
        group["d18O_scaled"] = slope_d18O*group['d18O']+intercept_d18O
        group["d17O_scaled"] = slope_d17O*group['d17O']+intercept_d17O
        group["Dp17O_scaled"] = Dp17O(group["d17O_scaled"], group["d18O_scaled"])

        # Print out the scaled values for the carbonate standards for each measurement period
        standards = ["DH11", "NBS18", "IAEA603"]
        for standard in standards:
            if standard in group["SampleName"].values:
                only_standard = group[group["SampleName"].str.contains(standard)].copy()
                only_standard[["d18O_AC", "d17O_AC", "Dp17O_AC"]] = only_standard.apply(lambda x: applyAFF(x["d18O_scaled"], x["d17O_scaled"], "calcite"), axis=1, result_type="expand")
                print_info(only_standard, "d18O_AC", "Dp17O_AC", standard); print("\t<--- scaled + AFF")

        # Assign colors and markers to samples
        categories = group["SampleName"].unique()
        markers = dict(zip(categories, ["o", "s", "D", "v", "^",
                                        "<", ">", "p", "P", "*"]*4))
        colors = dict(zip(categories, plt.cm.tab20(np.linspace(0, 1, 20))))

        # Figure: unscaled Dp17O vs time

        _, ax = plt.subplots()

        for cat in categories:
            data = group[group["SampleName"] == cat]
            ax.scatter(data["dateTimeMeasured"], data["Dp17O"],
                       marker=markers[cat], fc=colors[cat], label=cat)
            if np.isnan(data["Dp17OError"]).any() == False:
                plt.errorbar(group["dateTimeMeasured"], group["Dp17O"],
                             yerr=group["Dp17OError"],
                             fmt="none", color="#cacaca", zorder=0)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        ax.set_title(f"Measurement period: {period}")
        ax.set_ylabel("$\Delta\prime^{17}$O (ppm, unscaled CO$_2$)")
        ax.set_xlabel("Measurement date")
        ax.text(0.02, 0.98, SuppFig[FigNum], size=14, ha="left", va="top",
                transform=ax.transAxes, fontweight="bold")

        plt.savefig(os.path.join(sys.path[0], f"{project} Figure S1{SuppFig[FigNum]}"))
        plt.close()

        # Exclude the standards from the exported dataframe
        group = group[~group["SampleName"].str.contains("heavy|light|NBS|DH11|IAEA")]
        df_samples = pd.concat([df_samples, group])

        FigNum += 1

    return df_samples


# This function averages the scaled data from multiple measurement periods
def average_data(df):
    # Calculate the mean values from the replicate measurements
    df = df.loc[:, ["SampleName", "d18O_scaled", "d17O_scaled", "Dp17O_scaled"]]
    df_mean = df.groupby('SampleName').mean().reset_index()
    df_mean = df_mean.rename(columns={'d18O_scaled': 'd18O_CO2', 'd17O_scaled': 'd17O_CO2', 'Dp17O_scaled': 'Dp17O_CO2'})

    # Calculate the standard deviation from the replicate measurements
    df_std = df.groupby('SampleName').std().reset_index()
    df_std = df_std.rename(columns={'d18O_scaled': 'd18O_error', 'd17O_scaled': 'd17O_error', 'Dp17O_scaled': 'Dp17O_error'})

    dfMerged = df_mean.merge(df_std, on='SampleName')
    dfMerged['Replicates'] = df.groupby('SampleName').size().reset_index(name='counts')['counts']
    df = dfMerged

    return df


# Here we go!

# Scale the data
df = scaleData(pd.read_csv(os.path.join(sys.path[0], "DP Table S1.csv")), "DP")

# Average the data
df_avg = average_data(df)

# Merge data with sample info
info = pd.read_csv(os.path.join(sys.path[0], "DP Table 2.csv"))
df_avg = df_avg.merge(info, on='SampleName')

# Apply acid fractionation correction
df_avg[["d18O_AC", "d17O_AC", "Dp17O_AC"]] = df_avg.apply(lambda x: applyAFF(x["d18O_CO2"], x["d17O_CO2"], x["Mineralogy"]), axis=1, result_type="expand")

# Print the averaged data
print("All sample replicates averaged:")
print(df_avg.round({"Dp17O_CO2": 0, "Dp17O_error": 0, "Dp17O_AC": 0}).round(3))

# Assign sample info and export CSV
df_avg.to_csv(os.path.join(sys.path[0], "DP Table S2.csv"), index=False)
df_avg.to_excel(os.path.join(sys.path[0], "DP Table S2.xlsx"), index=False)


################ Create Figure S1 c-d ############
fig, (ax1, ax2) = plt.subplots(1, 2)

# Assign colors and markers to samples
df.sort_values(by="SampleName", inplace=True)
categories = df["SampleName"].unique()
markers = dict(zip(categories, ["o", "s", "D", "^", "v", "X", "P", "*", "o",
               "s", "D", "^", "v", "X", "P", "*", "o", "s", "D", "^", "v", "X", "P", "*"]))
colors = dict(zip(categories, plt.cm.tab20(np.linspace(0, 1, len(categories)))))


# Subplot a: scaled sample replicates
for cat in categories:
    data = df[df["SampleName"] == cat]
    ax1.scatter(prime(data["d18O_scaled"]), data["Dp17O_scaled"],
                marker=markers[cat], fc=colors[cat], label=cat)
    ax1.errorbar(prime(df["d18O_scaled"]), df["Dp17O_scaled"],
                 yerr=df["Dp17OError"],
                 xerr=df["d18OError"],
                 fmt="none", color="#cacaca", zorder=0)


# Axis properties
ylim = ax1.get_ylim()
xlim = ax1.get_xlim()
ax1.set_ylabel("$\Delta^{\prime 17}$O (ppm, CO$_2$)")
ax1.set_xlabel("$\delta^{\prime 18}$O (‰, VSMOW, CO$_2$)")
ax1.text(0.02, 0.98, "c", size=14, ha="left", va="top",
         transform=ax1.transAxes, fontweight="bold")

# Subplot B: scaled sample averages
for cat in categories:
    data = df_avg[df_avg["SampleName"] == cat]
    ax2.scatter(prime(data["d18O_CO2"]), data["Dp17O_CO2"],
                marker=markers[cat], fc=colors[cat], label=cat)
    ax2.errorbar(prime(df_avg["d18O_CO2"]), df_avg["Dp17O_CO2"],
                 yerr=df_avg["Dp17O_error"],
                 xerr=df_avg["d18O_error"],
                 fmt="none", color="#cacaca", zorder=0)

ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Axis properties
ax2.set_ylim(ylim)
ax2.set_xlim(xlim)
ax2.set_ylabel("$\Delta\prime^{17}$O (ppm, CO$_2$)")
ax2.set_xlabel("$\delta\prime^{18}$O (‰, VSMOW, CO$_2$)")
ax2.text(0.02, 0.98, "d", size=14, ha="left", va="top",
         transform=ax2.transAxes, fontweight="bold")

plt.savefig(os.path.join(sys.path[0], "DP Figure S1cd"))
plt.close("all")

subprocess.run([sys.executable, os.path.join(sys.path[0], "combineImage.py")])
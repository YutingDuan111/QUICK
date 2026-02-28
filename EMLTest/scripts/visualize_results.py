#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('../test_grids/grid_results.csv')

baseline_energy = df[(df["RadialPoints"] == 99) & (df["AngularPoints"] == 302)]["TotalEnergy"].values[0]
df["EnergyDeviation_mEh"] = (df["TotalEnergy"] - baseline_energy) * 1000
# Filter out points with deviation larger than 1e-2 Hartree
#MAX_DEVIATION = 10
#df_filtered = df[df["EnergyDeviation_mEh"].abs() <= MAX_DEVIATION].copy()

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
func_label = "PSB3 BLYP 6-31G\ncutoff=1.0e-9 denserms=1.0e-6"

df_r50 = df[df["RadialPoints"] == 50].sort_values("AngularPoints")
df_a194 = df[df["AngularPoints"] == 194].sort_values("RadialPoints")

#df_r50 = df_filtered[df_filtered["RadialPoints"] == 50].sort_values("AngularPoints")
#df_a194 = df_filtered[df_filtered["AngularPoints"] == 194].sort_values("RadialPoints")


axes[0].plot(df_r50["AngularPoints"], df_r50["EnergyDeviation_mEh"], marker="o")
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, label='Baseline (99/302)')
axes[0].set_xlabel("AngularPoints")
axes[0].set_ylabel("TotalEnergy (mEh)")
axes[0].set_title("Fixed RadialPoints = 50")
axes[0].grid(True)
axes[0].text(0.5, 0.95, func_label, transform=axes[0].transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1].plot(df_a194["RadialPoints"], df_a194["EnergyDeviation_mEh"], marker="o")
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1, label='Baseline (99/302)')
axes[1].set_xlabel("RadialPoints")
axes[1].set_ylabel("TotalEnergy (mEh)")
axes[1].set_title("Fixed AngularPoints = 194")
axes[1].grid(True)
axes[1].text(0.5, 0.95, func_label, transform=axes[1].transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../test_grids/eml_grid.png')
#plt.show()

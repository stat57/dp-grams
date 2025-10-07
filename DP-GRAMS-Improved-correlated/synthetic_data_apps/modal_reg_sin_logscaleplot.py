import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- visual style ----
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({"font.size": 12})

# -------------------------------
# Load CSV
# -------------------------------
results_dir = "results/modal_regression_sin"
csv_file = os.path.join(results_dir, "privacy_utility_modal_sin.csv")
df = pd.read_csv(csv_file)

# -------------------------------
# Parameters
# -------------------------------
n_values = sorted(df["n"].unique())
palette = sns.color_palette("Set2", len(n_values))

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8,6))

for i, n in enumerate(n_values):
    subset = df[df["n"] == n].sort_values("epsilon")  # sort by epsilon
    eps_subset = subset["epsilon"].values
    mse_subset = subset["DP-PMS MSE mean"].values
    mse_errs = subset["DP-PMS MSE SE"].values
    pms_mse_val = subset["PMS MSE"].iloc[0]

    # DP-PMS with error bars
    plt.errorbar(eps_subset, mse_subset, yerr=mse_errs,
                 marker='o', linestyle='-', linewidth=2, markersize=7,
                 capsize=3, label=f"DP-PMS (n={n})", color=palette[i])

    # PMS baseline
    plt.hlines(pms_mse_val, eps_subset.min(), eps_subset.max(),
               colors=palette[i], linestyles='dashed', linewidth=1.5,
               label=f"PMS baseline (n={n})")

plt.xscale('log')  # <---- log scale for ε
plt.xlabel("Privacy budget ε")
plt.ylabel("MSE")
plt.title("Privacy–Utility Tradeoff (Sinusoidal Modal Regression)")
plt.grid(axis='y', linestyle="--", alpha=0.4)
plt.legend(ncol=1, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir,"privacy_utility_modal_sin_log.png"), dpi=300)
plt.show()

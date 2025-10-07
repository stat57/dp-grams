import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import numpy as np

# ---- visual style ----
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

results_dir = "results/modal_regression"
csv_file = os.path.join(results_dir, "privacy_utility_results.csv")

# Load CSV
data = []
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "n": int(row["n_samples"]),
            "eps": float(row["epsilon"]),
            "dp_mse": float(row["DP_mse_mean"]),
            "dp_se": float(row["DP_mse_se"]),
            "pms_mse": float(row["PMS_mse"])
        })

# Unique n values
n_values = sorted(list(set(d["n"] for d in data)))

palette = sns.color_palette("Set2", len(n_values))
plt.figure(figsize=(8,6))

for i, n in enumerate(n_values):
    subset = [d for d in data if d["n"] == n]
    eps_vals = [d["eps"] for d in subset]
    dp_mse_vals = [d["dp_mse"] for d in subset]
    dp_se_vals = [d["dp_se"] for d in subset]
    pms_mse_val = subset[0]["pms_mse"]

    # DP-PMS curve
    plt.errorbar(eps_vals, dp_mse_vals, yerr=dp_se_vals,
                 marker='o', linestyle='-', linewidth=2, markersize=7,
                 capsize=3, label=f"DP-PMS (n={n})", color=palette[i])

    # PMS baseline
    plt.hlines(pms_mse_val, eps_vals[0], eps_vals[-1],
               colors=palette[i], linestyles='dashed', linewidth=1.5,
               label=f"PMS baseline (n={n})")

plt.xscale('log')  # <-- log scale for x-axis
# plt.yscale('log')
plt.xlabel("Privacy budget ε (log scale)")
plt.ylabel("MSE")
plt.title("Privacy–Utility Tradeoff in Modal Regression: DP-PMS vs PMS")
plt.grid(axis='y', linestyle="--", alpha=0.4)
plt.legend(ncol=1, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "privacy_utility_modal_logx.png"), dpi=300, bbox_inches="tight")
plt.show()

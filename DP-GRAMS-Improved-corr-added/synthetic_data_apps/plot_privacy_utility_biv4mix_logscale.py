# plot_privacy_utility_biv4mix_logscale.py
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Config
# -------------------------------
results_dir = "results/bivariate_4mix"

# Style
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

# -------------------------------
# Load data
# -------------------------------
epsilon_files = [f for f in os.listdir(results_dir) if f.startswith("privacy_vs_epsilon_n") and f.endswith(".csv")]
epsilon_files.sort()

dp_mse_eps_dict = {}
n_samples_list = []

for fname in epsilon_files:
    match = re.search(r"n(\d+)", fname)
    if not match:
        continue  # skip any files that don't match pattern
    n_val = int(match.group(1))

    if n_val == 500:  
        continue  # 🚨 skip n=500

    csv_path = os.path.join(results_dir, fname)
    df = pd.read_csv(csv_path)

    dp_means = df["DP_mean"].values
    dp_ses = df["DP_se"].values
    ms_mean = df["MS_mean"].iloc[0]
    ms_se = df["MS_se"].iloc[0]

    dp_mse_eps_dict[n_val] = (ms_mean, ms_se, dp_means, dp_ses, df["epsilon"].values)
    n_samples_list.append(n_val)

# -------------------------------
# Privacy–Utility Plot (log scale ε)
# -------------------------------
palette = sns.color_palette("Set2", len(n_samples_list))
plt.figure(figsize=(8,6))

for i, n in enumerate(sorted(n_samples_list)):
    if n==500:
        continue
    ms_mean, ms_se, dp_means, dp_ses, eps_values = dp_mse_eps_dict[n]

    # DP-GRAMS curve
    plt.errorbar(eps_values, dp_means, yerr=dp_ses,
                 marker='o', linestyle='-', linewidth=1.8,
                 markersize=6, capsize=3,
                 color=palette[i], label=f"DP-GRAMS n={n}")

    # MS baseline
    plt.hlines(ms_mean, min(eps_values), max(eps_values),
               colors=palette[i], linestyles='dashed', linewidth=1.2,
               label=f"MS n={n}")

plt.xscale("log")  # log scale for epsilon

# Force x-axis ticks at all epsilon values
all_eps = sorted(set(eps for eps_values in [dp_mse_eps_dict[n][4] for n in n_samples_list] for eps in eps_values))
plt.xticks(all_eps, labels=[f"{eps:.2g}" for eps in all_eps])

plt.xlabel("Privacy budget ε (log scale)")
plt.ylabel("MSE")
plt.title("Privacy–Utility Tradeoff: DP-GRAMS vs MS (Bivariate 4-Modal Gaussian Mixture)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(loc="upper right", frameon=True)

plt.tight_layout()
out_path = os.path.join(results_dir, "privacy_utility_biv4mix_log.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved log-scale privacy–utility plot to {out_path}")

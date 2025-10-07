# plot_privacy_utility_separate.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Paths
# ---------------------------
results_dir = "results/digits_centroid_comparison"
csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
out_dir = os.path.join(results_dir, "privacy_utility_plots")
os.makedirs(out_dir, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(csv_file)

# Metrics to plot
metrics = ["ARI", "NMI", "Silhouette", "MSE"]
yerr_cols = {
    "ARI": "Std_ARI",
    "NMI": "Std_NMI",
    "Silhouette": "Std_Silhouette",
    "MSE": "Std_MSE",
}

sns.set(style="whitegrid", context="talk")

# ---------------------------
# Generate one plot per metric
# ---------------------------
for metric in metrics:
    plt.figure(figsize=(7, 5))
    for alg in df["Algorithm"].unique():
        subset = df[df["Algorithm"] == alg]
        plt.errorbar(
            subset["Eps_modes"],
            subset[metric],
            yerr=subset[yerr_cols[metric]],
            marker="o" if alg == "DP-GRAMS-C" else "s",
            linestyle="-",
            capsize=3,
            label=alg,
        )
    plt.xscale("log")
    plt.xlabel(r"$\epsilon_{\mathrm{modes}}$")
    plt.ylabel(metric)
    plt.title(f"Privacy-Utility Tradeoff: {metric} for Digits Dataset.")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = os.path.join(out_dir, f"{metric.lower()}_privacy_utility.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {metric} plot to {save_path}")

print("All separate privacy-utility plots saved in:", out_dir)

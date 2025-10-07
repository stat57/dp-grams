import pandas as pd
import os, matplotlib.pyplot as plt, seaborn as sns

results_dir = "results/modal_regression"
dp_csv = os.path.join(results_dir, "dp_per_run_results.csv")
pms_csv = os.path.join(results_dir, "pms_results.csv")

# load data
dp_df = pd.read_csv(dp_csv)
pms_df = pd.read_csv(pms_csv)

# aggregate DP-PMS
dp_summary = (
    dp_df.groupby(["n_samples","epsilon"])
    .agg(
        DP_mse_mean=("dp_mse","mean"),
        DP_mse_se=("dp_mse",lambda x: x.std(ddof=1)/len(x)**0.5),
        DP_runtime_mean=("dp_runtime_s","mean"),
        DP_runtime_std=("dp_runtime_s","std")
    ).reset_index()
)

# merge PMS
final_df = dp_summary.merge(pms_df[["n_samples","PMS_mse","PMS_runtime"]],
                            on="n_samples", how="left")

final_csv = os.path.join(results_dir, "privacy_utility_final.csv")
final_df.to_csv(final_csv, index=False)
print(f"Saved merged results to {final_csv}")

# plot
plt.figure(figsize=(8,6))
palette = sns.color_palette("Set2", len(final_df["n_samples"].unique()))
for i, n in enumerate(sorted(final_df["n_samples"].unique())):
    subset = final_df[final_df["n_samples"]==n]
    plt.errorbar(subset["epsilon"], subset["DP_mse_mean"],
                 yerr=subset["DP_mse_se"], marker="o", linewidth=2,
                 label=f"DP-PMS (n={n})", color=palette[i])
    if not subset["PMS_mse"].isna().all():
        pms_val = subset["PMS_mse"].iloc[0]
        plt.hlines(pms_val, subset["epsilon"].min(), subset["epsilon"].max(),
                   colors=palette[i], linestyles="dashed",
                   linewidth=1.5, label=f"PMS baseline (n={n})")

plt.xscale('log')
plt.xlabel("Privacy budget ε")
plt.ylabel("MSE")
plt.title("Privacy–Utility Tradeoff in Modal Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "privacy_utility_combined.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# modal_reg_sin_groundtruth_final_full.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import time
from PMS import partial_mean_shift
from dp_pms import dp_pms
from statsmodels.nonparametric.smoothers_lowess import lowess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_scripts.mode_matching_mse import mode_matching_mse
import csv

# ---- visual style ----
sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

# -------------------------------
# Parameters
# -------------------------------
sigma = 0.15
n_base = 5000
results_dir = "results/modal_regression_sin"
os.makedirs(results_dir, exist_ok=True)
n_runs = 20
eps_values = [0.01, 0.05, 0.1, 0.5, 1]
n_values = [500, 1000, 2000, 5000]

# Reduce mesh size for speed (optional)
mesh_reduce_factor = 0.2  

# -------------------------------
# True modal function
# -------------------------------
def true_modal(X):
    n = len(X)//2
    Y_true = np.concatenate([1.5 + 0.5*np.sin(3*np.pi*X[:n]),
                             0.5*np.sin(3*np.pi*X[n:])])
    return Y_true

# -------------------------------
# Data generator
# -------------------------------
def generate_sin_data(n, sigma, seed=None):
    rng = np.random.default_rng(seed)
    X1 = rng.uniform(0,1,n//2)
    X2 = rng.uniform(0,1,n - n//2)
    X = np.concatenate([X1,X2])
    Y1 = rng.normal(1.5 + 0.5*np.sin(3*np.pi*X1), sigma, n//2)
    Y2 = rng.normal(0.5*np.sin(3*np.pi*X2), sigma, n - n//2)
    Y = np.concatenate([Y1,Y2])
    Y = Y / (np.std(Y)/np.std(X))  # normalize
    return X, Y

# -------------------------------
# Base dataset for visual comparison
# -------------------------------
eps_baseline = 1.0
X_base, Y_base = generate_sin_data(n_base, sigma, seed=42)
Y_true_base = true_modal(X_base)

# PMS
t0 = time.time()
Y_pms = partial_mean_shift(X_base, Y_base, T=7)
pms_time = time.time() - t0

# DP-PMS baseline (ε=1)
t0 = time.time()
modes_per_mesh, mesh_used = dp_pms(
    X_base, Y_base, mesh_X=X_base, epsilon=eps_baseline, delta=1e-5,
    rng=np.random.default_rng(42), clip_multiplier=0.01*mesh_reduce_factor
)
dpms_time_raw = time.time() - t0

X_dpms_exp = np.concatenate([[mesh_used[i]]*len(m) for i,m in enumerate(modes_per_mesh) if len(m)>0])
Y_dpms_exp = np.concatenate([m for m in modes_per_mesh if len(m)>0])
t0 = time.time()
Y_dpms_refined = partial_mean_shift(X_dpms_exp, Y_dpms_exp, T=int(np.log(len(Y_dpms_exp))))
dpms_time_refined = time.time() - t0 + dpms_time_raw

# LOWESS reference
Y_lowess = lowess(Y_base, X_base, frac=0.2, return_sorted=False)

# -------------------------------
# Visual comparison panel
# -------------------------------
fig, axes = plt.subplots(1,3,figsize=(18,5), sharex=True, sharey=True)

axes[0].scatter(X_base,Y_base,color='skyblue',alpha=0.5,s=15,label='Data')
axes[0].scatter(X_base,Y_pms,color='green',s=25,marker='D',label='PMS')
axes[0].set_title("Vanilla PMS"); axes[0].set_xlabel("X"); axes[0].set_ylabel("Y"); axes[0].legend()

axes[1].scatter(X_base,Y_base,color='skyblue',alpha=0.5,s=15,label='Data')
axes[1].scatter(X_base,Y_dpms_refined,color='purple',s=25,marker='X',label='DP-PMS')
axes[1].set_title(f"DP-PMS (ε={eps_baseline})"); axes[1].set_xlabel("X"); axes[1].legend()

axes[2].scatter(X_base,Y_base,color='skyblue',alpha=0.5,s=15,label='Data')
axes[2].plot(np.sort(X_base), Y_lowess[np.argsort(X_base)], color='orange', linewidth=2, label='LOWESS')
axes[2].set_title("LOWESS"); axes[2].set_xlabel("X"); axes[2].legend()

for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0,1)

plt.suptitle("Modal Regression Comparisons", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(results_dir,"modal_reg_comparison_sin.png"), dpi=300)
plt.show()

# -------------------------------
# Privacy–Utility Tradeoff: MSE vs ε, multiple n
# -------------------------------
results = []

for n in n_values:
    X_n, Y_n = generate_sin_data(n, sigma, seed=42)
    Y_true_n = true_modal(X_n)

    # PMS baseline
    t0 = time.time()
    Y_pms = partial_mean_shift(X_n, Y_n, T=7)
    pms_time_n = time.time() - t0
    pms_modes_arr = np.hstack([X_n.reshape(-1,1), Y_pms.reshape(-1,1)])
    true_modes_arr = np.hstack([X_n.reshape(-1,1), Y_true_n.reshape(-1,1)])
    pms_mse = mode_matching_mse(true_modes_arr, pms_modes_arr)

    for eps in eps_values:
        dp_mses, dp_times = [], []
        for run in range(n_runs):
            seed = 1000 * n + 10 * int(eps*1000) + run
            X_run, Y_run = X_n.copy(), Y_n.copy()
            t0 = time.time()
            modes_run, mesh_run = dp_pms(
                X_run, Y_run, mesh_X=X_run, epsilon=eps, delta=1e-5,
                rng=np.random.default_rng(seed), clip_multiplier=0.01*mesh_reduce_factor
            )
            dp_times.append(time.time()-t0)
            if len(modes_run)==0 or all(len(m)==0 for m in modes_run):
                continue
            X_dpms_exp = np.concatenate([mesh_run[i]*np.ones(len(m)) for i,m in enumerate(modes_run) if len(m)>0])
            Y_dpms_exp = np.concatenate([m for m in modes_run if len(m)>0])
            t0 = time.time()
            Y_dpms_ref = partial_mean_shift(X_dpms_exp, Y_dpms_exp, T=int(np.log(len(Y_dpms_exp))))
            dp_times[-1] += time.time()-t0
            est_modes = np.hstack([X_dpms_exp.reshape(-1,1), Y_dpms_ref.reshape(-1,1)])
            true_modes_arr = np.hstack([X_dpms_exp.reshape(-1,1), true_modal(X_dpms_exp).reshape(-1,1)])
            dp_mses.append(mode_matching_mse(true_modes_arr, est_modes))

        if dp_mses:
            dp_mean = np.mean(dp_mses)
            dp_se = np.std(dp_mses, ddof=1)/np.sqrt(len(dp_mses))
            time_mean = np.mean(dp_times)
            time_se = np.std(dp_times, ddof=1)/np.sqrt(len(dp_times))
        else:
            dp_mean, dp_se, time_mean, time_se = np.nan, np.nan, np.nan, np.nan

        results.append([n, eps, dp_mean, dp_se, pms_mse, pms_time_n, dp_mean, time_mean, time_se])

# -------------------------------
# Save results to CSV
# -------------------------------
csv_file = os.path.join(results_dir,"privacy_utility_modal_sin.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n","epsilon","DP-PMS MSE mean","DP-PMS MSE SE","PMS MSE","PMS runtime(s)",
                     "DP-PMS runtime mean(s)","DP-PMS runtime SE(s)"])
    for row in results:
        writer.writerow(row)

print("All results saved to:", csv_file)

# -------------------------------
# Privacy–Utility Plot
# -------------------------------
plt.figure(figsize=(8,6))
palette = sns.color_palette("Set2", len(n_values))

for i, n in enumerate(n_values):
    eps_subset = [r[1] for r in results if r[0]==n]
    mse_subset = [r[2] for r in results if r[0]==n]
    mse_errs = [r[3] for r in results if r[0]==n]
    pms_mse_val = results[i*len(eps_values)][4]

    plt.errorbar(eps_subset, mse_subset, yerr=mse_errs,
                 marker='o', linestyle='-', linewidth=2, markersize=7,
                 capsize=3, label=f"DP-PMS (n={n})", color=palette[i])
    plt.hlines(pms_mse_val, eps_subset[0], eps_subset[-1],
               colors=palette[i], linestyles='dashed', linewidth=1.5,
               label=f"PMS baseline (n={n})")

plt.xlabel("Privacy budget ε")
plt.ylabel("MSE")
plt.title("Privacy-Utility Tradeoff")
plt.grid(axis='y', linestyle="--", alpha=0.4)
plt.legend(ncol=1, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir,"privacy_utility_modal_sin_singleplot.png"), dpi=300)
plt.show()

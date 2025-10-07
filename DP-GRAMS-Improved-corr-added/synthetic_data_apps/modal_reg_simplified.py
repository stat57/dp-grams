# modal_reg_parallel_simplified.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os, sys
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import simplified DP-PMS and PMS
from dp_pms_simplified import dp_pms_simple
from PMS import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse
from statsmodels.nonparametric.smoothers_lowess import lowess

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
# Utilities
# -------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------
# Data generation (modal sine-like mixture)
# -------------------------------
true_modes_list = [3.0, 2.0, 1.0]

def generate_modal_data(n, sd=0.2, seed=None):
    rng = np.random.default_rng(seed)
    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2

    x1 = rng.uniform(0, 0.5, n1)
    x2 = rng.uniform(0.4, 0.7, n2)
    x3 = rng.uniform(0.6, 1.0, n3)

    y1 = rng.normal(true_modes_list[0], sd, n1)
    y2 = rng.normal(true_modes_list[1], sd, n2)
    y3 = rng.normal(true_modes_list[2], sd, n3)

    X = np.concatenate([x1, x2, x3])
    Y = np.concatenate([y1, y2, y3])
    return X, Y

# -------------------------------
# Experiment settings
# -------------------------------
results_dir = "results/modal_regression_simplified"
ensure_dir(results_dir)

n_values = [100, 500, 1000, 2000]
eps_values = [0.1, 0.2, 0.5, 1.0]
n_runs = 20
MAX_WORKERS = min(8, os.cpu_count() or 8)

perrun_csv = os.path.join(results_dir, "dp_per_run_results.csv")
if not os.path.exists(perrun_csv):
    with open(perrun_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples","epsilon","run_idx","seed","dp_mse","dp_runtime_s","timestamp"])

# -------------------------------
# DP-PMS worker (uses dp_pms_simple)
# -------------------------------
def dp_pms_worker(X, Y, epsilon, delta, seed, T_iter=None, verbose=False):
    """
    Worker that runs dp_pms_simple on (X,Y), refines with PMS and returns results.
    Returns: (seed, runtime_s, Y_refined, X_mesh)
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    n = len(X)
    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2, n))))

    # Use Y as mesh_points (one candidate per sample). dp_pms_simple returns Y_new (length n).
    Y_dp = dp_pms_simple(
        X, Y,
        mesh_points=Y.copy(),
        epsilon=float(epsilon),
        delta=float(delta),
        T=int(T_use),
        rng=rng,
        verbose=verbose
    )

    runtime = time.perf_counter() - t0
    return int(seed), float(runtime), Y_dp, X.copy()

# -------------------------------
# Main experiment
# -------------------------------
n_base = 500
eps_baseline = 1 

def main():
    print(f"[{now_str()}] Starting simplified modal regression experiments")
    # -----------------------
    # Baseline for debugging
    # -----------------------
    Xb, Yb = generate_modal_data(n_base, sd=0.2, seed=42)

    # PMS baseline
    t0 = time.perf_counter()
    Y_pms_truth = partial_mean_shift(Xb, Yb, mesh_points=None, T=int(np.log(n_base)))
    pms_time = time.perf_counter() - t0
    print(f"[{now_str()}] PMS baseline computed (n={n_base}), time={pms_time:.4f}s")

    # DP-PMS baseline (large epsilon)
    t0 = time.perf_counter()
    Y_dp_baseline = dp_pms_simple(Xb, Yb, mesh_points=Yb.copy(), 
                                  epsilon=eps_baseline, delta=1e-5, 
                                  T=int(np.log(n_base)), 
                                  rng=np.random.default_rng(42), verbose=True)
    dp_pms_time = time.perf_counter() - t0
    print(f"[{now_str()}] DP-PMS baseline (near-nonprivate) done, time={dp_pms_time:.4f}s")

    # LOWESS for visualization
    Y_lowess = lowess(Yb, Xb, frac=0.2, return_sorted=False)

    # Save baseline stats
    stats_text = (
        f"Baseline (n={n_base})\n"
        f"PMS runtime: {pms_time:.4f} s\n"
        f"DP-PMS runtime: {dp_pms_time:.4f} s\n"
    )
    save_text(os.path.join(results_dir, "stats_baseline.txt"), stats_text)

    # Plot baseline comparison
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True, sharey=True)
    axes[0].scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    axes[0].scatter(Xb, Y_pms_truth, color='#1f77b4', s=25, marker='D', label='PMS')
    axes[0].plot(np.sort(Xb), Y_lowess[np.argsort(Xb)], color='orange', linewidth=2, label='LOWESS')
    axes[0].set_title('PMS vs LOWESS'); axes[0].legend()

    axes[1].scatter(Xb, Yb, color='skyblue', alpha=0.5, s=15, label='Data')
    axes[1].scatter(Xb, Y_dp_baseline, color='#d95f02', s=35, marker='X', label=f'DP-PMS (ε={eps_baseline})')
    axes[1].set_title('DP-PMS'); axes[1].legend()

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0,1); ax.set_ylim(0,4)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.suptitle('Modal Regression Comparison in 3-Component Mixture Data', fontsize=16, y=0.97)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "baseline_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"[{now_str()}] Baseline figure saved")

    # ===========================================================
    # Privacy–Utility experiments (parallel)
    # ===========================================================
    results = []
    stats_summary = "=== Privacy–Utility Experiments ===\n"

    for n in n_values:
        print(f"\n[{now_str()}] Running experiments for n={n}")
        X_data, Y_data = generate_modal_data(n, sd=0.2, seed=123)

        # PMS ground truth estimate (use as reference)
        t0 = time.perf_counter()
        Y_pms_est = partial_mean_shift(X_data, Y_data, mesh_points=None, T=int(np.log(n)))
        pms_time_local = time.perf_counter() - t0
        pms_mse = mode_matching_mse(np.column_stack([X_data, Y_pms_est]), np.column_stack([X_data, Y_pms_est]))
        print(f"[{now_str()}] PMS computed for n={n}, time={pms_time_local:.4f}s")

        for eps in eps_values:
            print(f"[{now_str()}] DP runs: n={n}, eps={eps}")
            seeds = [1000*n + 10*int(eps*10) + run for run in range(n_runs)]
            dp_mses = []
            dp_times = []

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(dp_pms_worker, X_data, Y_data, float(eps), 1e-5, seed): (idx, seed)
                           for idx, seed in enumerate(seeds)}

                for fut in as_completed(futures):
                    run_idx, seed = futures[fut]
                    try:
                        seed_ret, runtime_ret, Y_refined, mesh_ret = fut.result()
                    except Exception as e:
                        print(f"[{now_str()}] ERROR run {run_idx} (seed={seed}): {e}")
                        continue

                    est_modes = np.column_stack([mesh_ret, Y_refined])
                    dp_mse = mode_matching_mse(np.column_stack([X_data, Y_pms_est]), est_modes)
                    dp_mses.append(dp_mse)
                    dp_times.append(runtime_ret)

                    with open(perrun_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([n, eps, run_idx, seed_ret, dp_mse, runtime_ret, now_str()])

                    print(f"[{now_str()}] Completed run {run_idx} (seed={seed_ret}); mse={dp_mse:.4f}; time={runtime_ret:.2f}s")

            # summarize per (n, eps)
            if len(dp_mses) == 0:
                print(f"[{now_str()}] No successful runs for n={n}, eps={eps}")
                continue

            dp_mean = float(np.mean(dp_mses))
            dp_se = float(np.std(dp_mses, ddof=1) / np.sqrt(len(dp_mses)))
            dp_time_mean = float(np.mean(dp_times))
            dp_time_std = float(np.std(dp_times, ddof=1) if len(dp_times) > 1 else 0.0)

            results.append([n, eps, dp_mean, dp_se, pms_mse, pms_time_local, dp_time_mean, dp_time_std])
            stats_summary += (
                f"n={n}, eps={eps} | DP-MSE={dp_mean:.4f}±{dp_se:.4f}, "
                f"PMS-MSE={pms_mse:.4f}, PMS-time={pms_time_local:.2f}s, "
                f"DP-time={dp_time_mean:.2f}±{dp_time_std:.2f}s\n"
            )

    # save aggregated CSV and stats
    csv_path = os.path.join(results_dir, "privacy_utility_results_simplified.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples","epsilon","DP_mse_mean","DP_mse_se","PMS_mse","PMS_runtime","DP_runtime_mean","DP_runtime_std"])
        writer.writerows(results)

    save_text(os.path.join(results_dir, "stats_summary.txt"), stats_summary)
    print(f"[{now_str()}] Experiments finished. Results saved to {results_dir}")

    # Plot privacy-utility tradeoff
    if len(results) > 0:
        plt.figure(figsize=(8,6))
        palette = sns.color_palette("Set2", len(n_values))
        for i, n in enumerate(n_values):
            eps_subset = [r[1] for r in results if r[0] == n]
            mse_subset = [r[2] for r in results if r[0] == n]
            mse_errs = [r[3] for r in results if r[0] == n]
            pms_mse_val = next((r[4] for r in results if r[0] == n), np.nan)
            if len(eps_subset) > 0:
                plt.errorbar(eps_subset, mse_subset, yerr=mse_errs, marker='o', linestyle='-', linewidth=2, markersize=7, capsize=3, label=f"DP-PMS (n={n})", color=palette[i])
                if not np.isnan(pms_mse_val):
                    plt.hlines(pms_mse_val, min(eps_subset), max(eps_subset), colors=palette[i], linestyles='dashed', linewidth=1.5, label=f"PMS baseline (n={n})")
        plt.xlabel("Privacy budget ε")
        plt.xscale('log')
        plt.ylabel("MSE")
        plt.title("Privacy–Utility Tradeoff in 3-Component Mixture Data")
        plt.grid(axis='y', linestyle="--", alpha=0.4)
        plt.legend(ncol=1, loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "privacy_utility_modal_simplified.pdf"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"[{now_str()}] Privacy-utility figure saved")

if __name__ == "__main__":
    # For Windows compatibility when using concurrent futures + multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()

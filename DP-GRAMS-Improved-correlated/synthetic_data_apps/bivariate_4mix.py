# synthetic_data_apps/bivariate_4mix.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
import math
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_scripts.ms import mean_shift
from main_scripts.dp_grams import dp_grams
from main_scripts.merge import *
from main_scripts.bandwidth import silverman_bandwidth
from main_scripts.mode_matching_mse import mode_matching_mse

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

# -------------------------------
# Data generation
# -------------------------------
def generate_4corners(n_samples, seed=None):
    rng = np.random.default_rng(seed)
    means = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]])
    cov = np.eye(2)
    pts = [rng.multivariate_normal(m, cov, n_samples // 4) for m in means]
    return np.vstack(pts), means

# -------------------------------
# Single-run contour plot (with data points + density)
# -------------------------------
def contour_plot_single(data, true_modes, ms_modes, dp_modes, results_dir):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import gaussian_kde
    from matplotlib.lines import Line2D

    ensure_dir(results_dir)
    fig = plt.figure(figsize=(14,6))

    # 3D KDE surface
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    kde = gaussian_kde(data.T)
    x_min, x_max = data[:,0].min()-2, data[:,0].max()+2
    y_min, y_max = data[:,1].min()-2, data[:,1].max()+2
    xgrid, ygrid = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)
    ax3d.plot_surface(xgrid, ygrid, Z, cmap='Blues', alpha=0.8)
    ax3d.set_xlabel(r"$x_1$"); ax3d.set_ylabel(r"$x_2$"); ax3d.set_zlabel("Density", labelpad=10)
    ax3d.set_title("Estimated Density Surface for 4-Modal Gaussian Mixture")

    # 2D contour + modes
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.contourf(xgrid, ygrid, Z, levels=50, cmap="Blues", alpha=0.25)
    ax2d.contour(xgrid, ygrid, Z, levels=8, colors='k', linewidths=0.5, alpha=0.5)
    ax2d.scatter(data[:,0], data[:,1], s=20, c="0.2", alpha=0.5, edgecolors='none')

    def plot_modes(modes, color, marker):
        if len(modes) == 0: return
        ax2d.scatter(modes[:,0], modes[:,1], s=250, facecolor='white', edgecolor='none', alpha=0.9)
        ax2d.scatter(modes[:,0], modes[:,1], s=120, facecolor=color, edgecolor='k', linewidth=1, marker=marker)

    plot_modes(np.atleast_2d(true_modes), "#2ca02c", "*")
    plot_modes(np.atleast_2d(ms_modes), "#1f77b4", "X")
    plot_modes(np.atleast_2d(dp_modes), "#ff7f0e", "o")

    ax2d.set_xlabel(r"$x_1$"); ax2d.set_ylabel(r"x_2")
    ax2d.set_title("Mode Estimation: True vs MS vs DP-GRAMS")
    ax2d.set_aspect('equal', 'box'); ax2d.tick_params(axis='both', which='both', length=0)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker="*", color="#2ca02c", label="True modes", markeredgecolor="k", markersize=12, linestyle=""),
        Line2D([0],[0], marker="X", color="#1f77b4", label="MS", markersize=10, linestyle=""),
        Line2D([0],[0], marker="o", color="#ff7f0e", label="DP-GRAMS", markeredgecolor="k", markersize=10, linestyle="")
    ]
    ax2d.legend(handles=legend_handles, frameon=True, loc="upper right", fontsize=11)

    plt.tight_layout()
    outpath = os.path.join(results_dir, "contour_modes_3d.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved single-run 3D + contour plot to: {outpath}")

# -------------------------------
# Grid of 20 runs with data points + density
# -------------------------------
def contour_plot_grid(data, true_modes, ms_modes_all, dp_modes_all, results_dir,
                      nrows=4, ncols=5, figsize_per_subplot=(4.5,4.5)):
    from scipy.stats import gaussian_kde
    from matplotlib import gridspec
    from matplotlib.lines import Line2D

    ensure_dir(results_dir)
    total_slots = nrows * ncols
    nplots = len(ms_modes_all)

    kde = gaussian_kde(data.T)
    x_min, x_max = data[:,0].min()-2, data[:,0].max()+2
    y_min, y_max = data[:,1].min()-2, data[:,1].max()+2
    xgrid, ygrid = np.meshgrid(np.linspace(x_min, x_max, 300),
                               np.linspace(y_min, y_max, 300))
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])
    Z = np.reshape(kde(positions), xgrid.shape)

    fig_w = figsize_per_subplot[0]*ncols
    fig_h = figsize_per_subplot[1]*nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.15, hspace=0.18)

    for slot in range(total_slots):
        r = slot // ncols
        c = slot % ncols
        ax = fig.add_subplot(gs[r, c])

        if slot >= nplots:
            ax.axis("off")
            continue

        ax.contourf(xgrid, ygrid, Z, levels=50, cmap="Blues", alpha=0.2, zorder=0)
        ax.contour(xgrid, ygrid, Z, levels=6, colors='k', linewidths=0.5, alpha=0.4, zorder=1)
        ax.scatter(data[:,0], data[:,1], s=15, c="0.2", alpha=0.4, edgecolors='none', zorder=2)

        def plot_modes_subplot(modes, color, marker, zorder_base=3):
            if len(modes) == 0: return
            ax.scatter(modes[:,0], modes[:,1], s=200, facecolor='white', edgecolor='none', alpha=0.9, zorder=zorder_base)
            ax.scatter(modes[:,0], modes[:,1], s=90, facecolor=color, edgecolor='k', linewidth=0.8, marker=marker, zorder=zorder_base+1)

        plot_modes_subplot(np.atleast_2d(true_modes), "#2ca02c", "*")
        plot_modes_subplot(np.atleast_2d(ms_modes_all[slot]), "#1f77b4", "X")
        plot_modes_subplot(np.atleast_2d(dp_modes_all[slot]), "#ff7f0e", "o")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Run {slot+1}", fontsize=10)

    legend_handles = [
        Line2D([0],[0], marker="*", color="#2ca02c", label="True modes", markeredgecolor="k", markersize=12, linestyle=""),
        Line2D([0],[0], marker="X", color="#1f77b4", label="MS", markersize=10, linestyle=""),
        Line2D([0],[0], marker="o", color="#ff7f0e", label="DP-GRAMS", markeredgecolor="k", markersize=10, linestyle="")
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=True, fontsize=11)
    fig.suptitle("Mode Estimation Across 20 Runs: Bivariate 4-Modal Gaussian Mixture", fontsize=14, y=0.94)
    plt.tight_layout(rect=[0,0.03,1,0.95])

    outpath = os.path.join(results_dir, "contour_modes_grid.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved 20-run grid contour plot to: {outpath}")

# -------------------------------
# MSE vs n samples
# -------------------------------
def mse_vs_n_samples(n_samples_list, epsilon, delta, p,  n_runs, base_seed=42, results_dir="results/bivariate_4mix"):
    ms_mse_means, ms_mse_ses = [], []
    dp_mse_means, dp_mse_ses = [], []
    ms_time_means, ms_time_ses = [], []
    dp_time_means, dp_time_ses = [], []

    for idx, n_samples in enumerate(n_samples_list):
        ms_mses, dp_mses = [], []
        ms_times, dp_times = [], []
        
        data, true_modes = generate_4corners(n_samples, seed=base_seed + idx)
       
        h = silverman_bandwidth(data)

        for run in range(n_runs):
            run_seed = base_seed + idx*100 + run
            rng = np.random.default_rng(run_seed)

            t0 = time.perf_counter()
            ms_raw = mean_shift(data, T=int(np.log(n_samples)), bandwidth=h, p=p, seed=rng)
            ms_est = merge_modes(ms_raw)
            ms_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            _, dp_raw = dp_grams(X=data, epsilon=epsilon, delta=delta, T=None, p0=p, rng=rng, h=h)
            dp_est = merge_modes(dp_raw)
            dp_time = time.perf_counter() - t0

            ms_mses.append(mode_matching_mse(true_modes, ms_est))
            dp_mses.append(mode_matching_mse(true_modes, dp_est))
            ms_times.append(ms_time)
            dp_times.append(dp_time)

        ms_mse_means.append(float(np.mean(ms_mses)))
        ms_mse_ses.append(float(np.std(ms_mses, ddof=1) / math.sqrt(n_runs)))
        dp_mse_means.append(float(np.mean(dp_mses)))
        dp_mse_ses.append(float(np.std(dp_mses, ddof=1) / math.sqrt(n_runs)))

        ms_time_means.append(float(np.mean(ms_times)))
        ms_time_ses.append(float(np.std(ms_times, ddof=1) / math.sqrt(n_runs)))
        dp_time_means.append(float(np.mean(dp_times)))
        dp_time_ses.append(float(np.std(dp_times, ddof=1) / math.sqrt(n_runs)))

        print(f"[mse_vs_n] n={n_samples}: MS MSE={ms_mse_means[-1]:.6f}±{ms_mse_ses[-1]:.6f} | DP MSE={dp_mse_means[-1]:.6f}±{dp_mse_ses[-1]:.6f}")

    ensure_dir(results_dir)
    csv_path = os.path.join(results_dir, "privacy_vs_n.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples", "MS_mean", "MS_se", "DP_mean", "DP_se", "MS_time_mean", "MS_time_se", "DP_time_mean", "DP_time_se"])
        for i, n in enumerate(n_samples_list):
            writer.writerow([n,
                             ms_mse_means[i], ms_mse_ses[i],
                             dp_mse_means[i], dp_mse_ses[i],
                             ms_time_means[i], ms_time_ses[i],
                             dp_time_means[i], dp_time_ses[i]])
    print(f"Saved numeric privacy-vs-n results to {csv_path}")

    return ms_mse_means, ms_mse_ses, dp_mse_means, dp_mse_ses

# -------------------------------
# MSE vs epsilon
# -------------------------------
def mse_vs_epsilon_for_n_samples(n_samples_list, epsilon_values, delta, p,  n_runs, base_seed=42, results_dir="results/bivariate_4mix"):
    dp_mse_dict = {}

    for idx, n_samples in enumerate(n_samples_list):
        data, true_modes = generate_4corners(n_samples, seed=base_seed + idx)
       
        h = silverman_bandwidth(data)

        # MS baseline (no epsilon)
        ms_mses = []
        for run in range(n_runs):
            run_seed = base_seed + idx*100 + run
            rng = np.random.default_rng(run_seed)
            ms_raw = mean_shift(data, T=int(np.log(n_samples)), bandwidth=h, p=p, seed=rng)
            ms_est = merge_modes(ms_raw)
            ms_mses.append(mode_matching_mse(true_modes, ms_est))
        ms_mean = float(np.mean(ms_mses))
        ms_se = float(np.std(ms_mses, ddof=1)/math.sqrt(n_runs))

        dp_means, dp_ses = [], []
        dp_time_means, dp_time_ses = [], []

        for eps_idx, eps in enumerate(epsilon_values):
            dp_mses, dp_times = [], []
            for run in range(n_runs):
                run_seed = base_seed + idx*100 + run + eps_idx*1000
                rng = np.random.default_rng(run_seed)

                t0 = time.perf_counter()
                _, dp_raw = dp_grams(X=data, epsilon=eps, delta=delta, T=None, p0=p, rng=rng, h=h)
                dp_est = merge_modes(dp_raw)
                dp_time = time.perf_counter() - t0

                dp_mses.append(mode_matching_mse(true_modes, dp_est))
                dp_times.append(dp_time)

            dp_means.append(float(np.mean(dp_mses)))
            dp_ses.append(float(np.std(dp_mses, ddof=1)/math.sqrt(n_runs)))
            dp_time_means.append(float(np.mean(dp_times)))
            dp_time_ses.append(float(np.std(dp_times, ddof=1)/math.sqrt(n_runs)))

            print(f"[mse_vs_eps] n={n_samples}, eps={eps:.3f}: DP MSE={dp_means[-1]:.6f}±{dp_ses[-1]:.6f}")

        dp_mse_dict[n_samples] = (ms_mean, ms_se, dp_means, dp_ses)

        ensure_dir(results_dir)
        csv_path = os.path.join(results_dir, f"privacy_vs_epsilon_n{n_samples}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epsilon", "DP_mean", "DP_se", "DP_time_mean", "DP_time_se", "MS_mean", "MS_se"])
            for i, eps in enumerate(epsilon_values):
                writer.writerow([eps, dp_means[i], dp_ses[i], dp_time_means[i], dp_time_ses[i], ms_mean, ms_se])
        print(f"Saved numeric privacy-vs-epsilon results to {csv_path}")

    return dp_mse_dict

# -------------------------------
# Privacy–Utility plot 
# -------------------------------
def privacy_utility_plot(n_samples_list, epsilon_values,
                         ms_mse_n, ms_se_n, dp_mse_n, dp_se_n,
                         dp_mse_eps_dict, results_dir):
    ensure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(8,6))

    palette = sns.color_palette("Set2", len(n_samples_list))
    for i, n_samples in enumerate(n_samples_list):
        ms_mean, ms_se, dp_means, dp_ses = dp_mse_eps_dict[n_samples]
        ax.errorbar(epsilon_values, dp_means, yerr=dp_ses,
                    marker='o', linestyle='-', linewidth=1.8,
                    markersize=6, label=f"DP-GRAMS n={n_samples}",
                    color=palette[i], capsize=3)
        ax.hlines(ms_mean, epsilon_values[0], epsilon_values[-1],
                  colors=palette[i], linestyles='dashed', linewidth=1.2,
                  label=f"MS n={n_samples}")

    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("MSE")
    ax.set_title("Privacy–Utility Tradeoff: DP-GRAMS vs MS")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    outpath = os.path.join(results_dir, "privacy_utility_pretty.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved simplified privacy-utility plot to: {outpath}")

# -------------------------------
# Main simulation
# -------------------------------
def main():
    epsilon, delta = 1e0, 1e-6
    p = 0.1
    n_runs = 20
    results_dir = "results/bivariate_4mix"
    ensure_dir(results_dir)

    base_seed = 42
    n_samples_base = 4000
    data, true_modes = generate_4corners(n_samples_base, seed=base_seed)
    
    h = silverman_bandwidth(data) 
    print(f"[main] Silverman bandwidth h = {h:.6f}")

    T = int(math.ceil(math.log(max(2, n_samples_base)))) 

    ms_modes_all, dp_modes_all = [], []
    ms_mses, dp_mses = [], []
    ms_times, dp_times = [], []

    for run in range(n_runs):
        run_seed = base_seed + run
        rng = np.random.default_rng(run_seed)

        t0 = time.perf_counter()
        ms_raw = mean_shift(data, T=T, bandwidth=h, p=p, seed=rng)
        ms_est = merge_modes(ms_raw)
        ms_times.append(time.perf_counter() - t0)
        ms_modes_all.append(ms_est)

        t0 = time.perf_counter()
        _, dp_raw = dp_grams(X=data, epsilon=epsilon, delta=delta, T=T, h=h, p0=p,rng=rng)
        dp_est = merge_modes(dp_raw)
        dp_times.append(time.perf_counter() - t0)
        dp_modes_all.append(dp_est)

        ms_mse = mode_matching_mse(true_modes, ms_est)
        dp_mse = mode_matching_mse(true_modes, dp_est)
        ms_mses.append(ms_mse)
        dp_mses.append(dp_mse)

        print(f"[run {run+1}/{n_runs}] MS modes={len(ms_est)} | DP modes={len(dp_est)} | MS MSE={ms_mse:.6f} | DP MSE={dp_mse:.6f}")

    # Save stats and all modes
    stats = (
        f"n_samples: {n_samples_base}\n"
        f"dimension d: {data.shape[1]}\n"
        f"T (MS iterations): 20\n"
        f"T (DP iterations chosen): {T}\n"
        f"privacy epsilon: {epsilon}\n"
        f"privacy delta: {delta}\n"
        f"bandwidth h (Silverman on data): {h:.6f}\n\n"
        "=== Accuracy summary ===\n"
        f"MS MSE mean±std: {np.mean(ms_mses):.6f} ± {np.std(ms_mses):.6f}\n"
        f"DP-GRAMS MSE mean±std: {np.mean(dp_mses):.6f} ± {np.std(dp_mses):.6f}\n"
    )
    stats += (
    f"\n=== Runtime summary (seconds) ===\n"
    f"MS mean±std: {np.mean(ms_times):.4f} ± {np.std(ms_times):.4f}\n"
    f"DP-GRAMS mean±std: {np.mean(dp_times):.4f} ± {np.std(dp_times):.4f}\n"
)
    save_text(os.path.join(results_dir, "stats.txt"), stats)

    all_modes_text = ""
    for i in range(n_runs):
        all_modes_text += f"Run {i+1}\n"
        all_modes_text += "MS Modes:\n"
        all_modes_text += np.array2string(np.atleast_2d(ms_modes_all[i]), precision=6) + "\n"
        all_modes_text += f"MS MSE: {ms_mses[i]:.6f}\n"
        all_modes_text += "DP-GRAMS Modes:\n"
        all_modes_text += np.array2string(np.atleast_2d(dp_modes_all[i]), precision=6) + "\n"
        all_modes_text += f"DP MSE: {dp_mses[i]:.6f}\n\n"
    save_text(os.path.join(results_dir, "all_modes.txt"), all_modes_text)

    # Plots
    contour_plot_single(data, true_modes, ms_modes_all[0], dp_modes_all[0], results_dir)
    contour_plot_grid(data, true_modes, ms_modes_all, dp_modes_all, results_dir, ncols=5)

    # Privacy-utility plots
    n_samples_list = [500, 1000, 2000, 5000]
    epsilon_values = [0.1, 0.2, 0.5, 1, 5]

    ms_mse_n, ms_se_n, dp_mse_n, dp_se_n = mse_vs_n_samples(
        n_samples_list, epsilon, delta, p, n_runs, base_seed=base_seed, results_dir=results_dir
    )
    dp_mse_eps_dict = mse_vs_epsilon_for_n_samples(
        n_samples_list, epsilon_values, delta, p,  n_runs, base_seed=base_seed, results_dir=results_dir
    )
    privacy_utility_plot(n_samples_list, epsilon_values, ms_mse_n, ms_se_n, dp_mse_n, dp_se_n, dp_mse_eps_dict, results_dir)

    print("Done. Results saved to:", results_dir)

if __name__ == "__main__":
    main()

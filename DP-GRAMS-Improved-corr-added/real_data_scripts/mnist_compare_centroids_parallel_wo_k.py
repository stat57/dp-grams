# mnist_compare_centroids_parallel_full_saved.py

import numpy as np
import os
import sys
import time
import csv
import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dpms_cluster import dpms_private_rr
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative
from diffprivlib.models import KMeans as DPKMeans

# ---------------------------
# Parameters
# ---------------------------
merge_n_clusters = 10  # MNIST has 10 classes
ms_T = 10
ms_p = 0.1
eps_rr_high = 1e2
delta = 1e-5
p_seed = 0.1
base_rng_seed = 41
n_runs = 20   # number of repeats per epsilon
results_dir = "results/mnist_centroid_comparison"
os.makedirs(results_dir, exist_ok=True)
sns.set(style="whitegrid", context="talk")
eps_modes_list = [0.1, 0.5, 1, 2, 5, 10]
MAX_WORKERS = min(8, os.cpu_count() or 8)  # use 8 or available CPU count

# ---------------------------
# Helper functions
# ---------------------------
def compute_metrics(y_true, labels, X):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return ari, nmi, sil

def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def relabel_clusters(y_true, y_pred):
    """
    Robust relabeling that:
    - accepts arbitrary label values in y_pred (not necessarily 0..k-1)
    - works when #predicted clusters != #true clusters
    - returns new_labels aligned to y_true's label set (unique values in y_true)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    # build contingency matrix manually (rows: true labels, cols: predicted labels)
    true_to_idx = {lab: i for i, lab in enumerate(unique_true)}
    pred_to_idx = {lab: j for j, lab in enumerate(unique_pred)}

    cm = np.zeros((unique_true.size, unique_pred.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[true_to_idx[t], pred_to_idx[p]] += 1

    # assignment: maximize matches -> minimize negative matches
    row_ind, col_ind = linear_sum_assignment(-cm)

    # mapping from predicted label value -> assigned true label value
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[unique_pred[c]] = unique_true[r]

    # for any predicted labels not in mapping (more predicted clusters than true),
    # map them to the true label with which they have largest overlap
    for j, pred_label in enumerate(unique_pred):
        if pred_label not in mapping:
            best_row = cm[:, j].argmax()
            mapping[pred_label] = unique_true[best_row]

    # apply mapping to y_pred
    new_labels = np.array([mapping[p] for p in y_pred])
    return new_labels

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Single-run DPMS-C helper (executed in worker process)
# returns: (modes_pca, labels, (ARI,NMI,Sil), runtime_s, rr_p, seed)
# ---------------------------
def dpms_single_run(run, X, y_true, eps_modes, merge_n_clusters, base_seed=41):
    seed = base_seed + run
    t0 = time.time()
    print(f"[{now_str()}] [DPMS worker] start run={run} eps={eps_modes} seed={seed}")
    rng_run = np.random.default_rng(seed)
    modes_dpms_run, labels_dpms_run, rr_p = dpms_private_rr(
        data=X, epsilon_modes=eps_modes, epsilon_rr=eps_rr_high, delta=delta,
        p_seed=p_seed,
        rng=rng_run, k_est=merge_n_clusters
    )
    runtime = time.time() - t0
    labels_dpms_run = relabel_clusters(y_true, labels_dpms_run)
    run_metrics = compute_metrics(y_true, labels_dpms_run, X)
    print(f"[{now_str()}] [DPMS worker] end   run={run} eps={eps_modes} ARI={run_metrics[0]:.4f} time={runtime:.1f}s")
    return modes_dpms_run, labels_dpms_run, run_metrics, float(runtime), float(rr_p), int(seed)

# ---------------------------
# Single-run DP-KMeans helper (worker)
# returns: (modes_pca, labels, (ARI,NMI,Sil), runtime_s, seed)
# ---------------------------
def dpkm_single_run(run, X, y_true, eps_modes, merge_n_clusters, base_seed=41):
    seed = base_seed + run
    t0 = time.time()
    print(f"[{now_str()}] [DP-KMeans worker] start run={run} eps={eps_modes} seed={seed}")
    dp_kmeans_run = DPKMeans(n_clusters=merge_n_clusters, epsilon=eps_modes, random_state=seed)
    try:
        dp_kmeans_run.fit(X)
    except ValueError:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        dp_kmeans_run.fit(X, bounds=(mins, maxs))
    runtime = time.time() - t0
    labels_dpkm_run = dp_kmeans_run.labels_.astype(int) if hasattr(dp_kmeans_run, "labels_") else dp_kmeans_run.predict(X)
    labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run)
    run_metrics = compute_metrics(y_true, labels_dpkm_run, X)
    print(f"[{now_str()}] [DP-KMeans worker] end   run={run} eps={eps_modes} ARI={run_metrics[0]:.4f} time={runtime:.2f}s")
    return dp_kmeans_run.cluster_centers_, labels_dpkm_run, run_metrics, float(runtime), int(seed)

# ---------------------------
# Main
# ---------------------------
def main():
    print(f"[{now_str()}] Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw = mnist.data.astype(np.float32)
    y_true = mnist.target.astype(int)
    print(f"[{now_str()}] MNIST data shape: {X_raw.shape}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in np.unique(y_true)])

    # PCA 50D embedding (cache)
    pca_cache = os.path.join(results_dir, "mnist_pca_50d.npy")
    if os.path.exists(pca_cache):
        print(f"[{now_str()}] Loading cached PCA-50 embedding...")
        X = np.load(pca_cache)
        pca_50 = PCA(n_components=50, random_state=base_rng_seed)
        pca_50.fit(X_raw)
    else:
        print(f"[{now_str()}] Computing PCA-50 embedding...")
        pca_50 = PCA(n_components=50, random_state=base_rng_seed)
        X = pca_50.fit_transform(X_raw)
        np.save(pca_cache, X)
        print(f"[{now_str()}] PCA-50 embedding saved to: {pca_cache}")

    true_means = np.array([X[y_true == c].mean(axis=0) for c in np.unique(y_true)])

    # ---------------------------
    # Single clustering comparisons (keep as-is, but save results)
    # ---------------------------
    print(f"[{now_str()}] Running single clustering comparisons (Mean-Shift, DPMS-C, KMeans, DP-KMeans)...")
    rng_ms = np.random.default_rng(base_rng_seed)
    raw_ms_modes, ms_time = timed(mean_shift, X, None, ms_T, None, ms_p, rng_ms)
    ms_merged_modes = merge_modes_agglomerative(raw_ms_modes, n_clusters=merge_n_clusters, random_state=base_rng_seed)
    dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
    labels_ms = np.argmin(dists_ms, axis=1)
    labels_ms = relabel_clusters(y_true, labels_ms)

    kmeans = KMeans(n_clusters=merge_n_clusters, random_state=base_rng_seed, n_init=10)
    labels_km, km_time = timed(kmeans.fit_predict, X)
    modes_km = kmeans.cluster_centers_
    labels_km = relabel_clusters(y_true, labels_km)

    dp_kmeans = DPKMeans(n_clusters=merge_n_clusters, epsilon=1, random_state=base_rng_seed)
    try:
        dp_kmeans_out, dpkm_time = timed(dp_kmeans.fit, X)
    except ValueError:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        dp_kmeans_out, dpkm_time = timed(dp_kmeans.fit, X, bounds=(mins, maxs))
    # After fit, extract labels and centers
    labels_dpkm = dp_kmeans.labels_.astype(int) if hasattr(dp_kmeans, "labels_") else dp_kmeans.predict(X)
    modes_dpkm = dp_kmeans.cluster_centers_
    labels_dpkm = relabel_clusters(y_true, labels_dpkm)

    rng_dpms = np.random.default_rng(base_rng_seed)
    (modes_dpms, labels_dpms, rr_p), dpms_time = timed(
        dpms_private_rr,
        data=X,
        epsilon_modes=1,
        epsilon_rr=eps_rr_high,
        delta=delta,
        p_seed=p_seed,
        rng=rng_dpms,
        k_est=merge_n_clusters
    )
    labels_dpms = relabel_clusters(y_true, labels_dpms)

    algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
    labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
    modes_list = [ms_merged_modes, modes_dpms, modes_km, modes_dpkm]
    runtimes = [ms_time, dpms_time, km_time, dpkm_time]

    metrics = {}
    mse_centroids = {}
    for alg, labels, modes in zip(algorithms, labels_list, modes_list):
        metrics[alg] = compute_metrics(y_true, labels, X)
        modes_orig = pca_50.inverse_transform(modes)
        modes_scaled = scaler.transform(modes_orig)
        mse_centroids[alg] = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

    # Print and save single-run numeric results (CSV + TXT)
    single_csv = os.path.join(results_dir, "clustering_metrics_single_run.csv")
    with open(single_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "ARI", "NMI", "Silhouette", "MSE", "Runtime(s)"])
        for alg, rt in zip(algorithms, runtimes):
            writer.writerow([alg, metrics[alg][0], metrics[alg][1], metrics[alg][2], mse_centroids[alg], rt])
    print(f"[{now_str()}] Single-run clustering metrics saved to {single_csv}")

    single_txt = os.path.join(results_dir, "clustering_metrics_single_run.txt")
    with open(single_txt, "w") as f:
        for alg, rt in zip(algorithms, runtimes):
            vals = metrics[alg]
            f.write(f"{alg}: ARI={vals[0]:.4f}, NMI={vals[1]:.4f}, "
                    f"Silhouette={vals[2]:.4f}, MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n")
    print(f"[{now_str()}] Single-run clustering metrics (text) saved to {single_txt}")

    # Print results to console
    print("\nSingle-run metrics (printed):")
    for alg, rt in zip(algorithms, runtimes):
        vals = metrics[alg]
        print(f"{alg:<12} ARI={vals[0]:.4f}, NMI={vals[1]:.4f}, Silhouette={vals[2]:.4f}, MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s")

    # ---------------------------
    # Clustering comparison plot (same as before)
    # ---------------------------
    pca_2d = PCA(n_components=2, random_state=base_rng_seed)
    X_2d = pca_2d.fit_transform(X)
    true_means_2d = pca_2d.transform(true_means)
    modes_2d_list = [pca_2d.transform(m) for m in modes_list]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows x 2 cols

    palette = sns.color_palette("tab10", merge_n_clusters)
    global_handles, global_labels = [], []

    for ax, alg, labels_pred, modes_2d in zip(axes.flat, algorithms, labels_list, modes_2d_list):

        for i, color in enumerate(palette):
            sc = ax.scatter(X_2d[labels_pred == i, 0], X_2d[labels_pred == i, 1], c=[color], s=40, alpha=0.7)
            if len(global_handles) < merge_n_clusters:
                global_handles.append(sc)
                global_labels.append(f"Cluster {i}")
        true_sc = ax.scatter(true_means_2d[:, 0], true_means_2d[:, 1], marker="X", c="magenta", s=180, linewidths=2)
        if "True means" not in global_labels:
            global_handles.append(true_sc)
            global_labels.append("True means")
        modes_sc = ax.scatter(modes_2d[:, 0], modes_2d[:, 1], marker="X", c="blue", s=120, linewidths=2)
        if "Estimated modes" not in global_labels:
            global_handles.append(modes_sc)
            global_labels.append("Estimated modes")
        ax.set_title(alg, fontsize=16)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    legend = fig.legend(global_handles, global_labels, fontsize=10, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.93),
                        title="Cluster Assignments & Centroids")
    plt.setp(legend.get_title(), fontsize=12, fontweight="bold")
    fig.suptitle("Clustering Comparison on MNIST Dataset", fontsize=20, y=0.98)
    save_path_clusters = os.path.join(results_dir, "mnist_clustering_comparison.png")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path_clusters, dpi=300)
    plt.show()
    print(f"[{now_str()}] Clustering comparison saved to: {save_path_clusters}")

    # ---------------------------
    # Prepare per-run CSVs for DPMS and DP-KMeans (append mode, in case of restart)
    # ---------------------------
    dpms_perrun_csv = os.path.join(results_dir, "dpms_per_run_results.csv")
    dpkm_perrun_csv = os.path.join(results_dir, "dpkm_per_run_results.csv")

    dpms_header = ["Algorithm", "Eps_modes", "Run", "Seed", "ARI", "NMI", "Silhouette", "MSE", "Runtime_s", "rr_p", "timestamp"]
    dpkm_header = ["Algorithm", "Eps_modes", "Run", "Seed", "ARI", "NMI", "Silhouette", "MSE", "Runtime_s", "timestamp"]

    # create files with headers if not exist
    if not os.path.exists(dpms_perrun_csv):
        with open(dpms_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(dpms_header)
    if not os.path.exists(dpkm_perrun_csv):
        with open(dpkm_perrun_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(dpkm_header)

    # ---------------------------
    # Parallel DPMS-C and DP-KMeans for privacy-utility
    # ---------------------------
    dpms_metrics, dpkm_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}, {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpms_err, dpkm_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}, {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
    dpms_times_mean, dpms_times_std = [], []
    dpkm_times_mean, dpkm_times_std = [], []

    print(f"[{now_str()}] Starting parallel experiments: {len(eps_modes_list)} epsilons x {n_runs} runs each (DPMS & DP-KMeans).")
    for eps in eps_modes_list:
        print(f"\n[{now_str()}] === Running DPMS-C and DP-KMeans for eps={eps} ===")

        # DPMS-C (parallel)
        ari_list, nmi_list, sil_list, mse_list, runtimes_list = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(dpms_single_run, run, X, y_true, eps, merge_n_clusters, base_rng_seed): run for run in range(n_runs)}
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    modes_run, labels_run, run_metrics, runtime_run, rr_p_run, seed_run = future.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DPMS future for run {run_id} failed: {e}")
                    continue

                # compute MSE in pixel space (modes_run are in PCA space)
                modes_orig = pca_50.inverse_transform(modes_run)
                modes_scaled = scaler.transform(modes_orig)
                mse_run = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

                # append to lists for aggregation
                ari_list.append(run_metrics[0])
                nmi_list.append(run_metrics[1])
                sil_list.append(run_metrics[2])
                mse_list.append(mse_run)
                runtimes_list.append(runtime_run)

                # write per-run CSV row immediately (safe on crash)
                with open(dpms_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "DP-GRAMS-C", eps, run_id, seed_run,
                        float(run_metrics[0]), float(run_metrics[1]), float(run_metrics[2]),
                        float(mse_run), float(runtime_run), float(rr_p_run), now_str()
                    ])

                print(f"[{now_str()}] [DPMS main] wrote run={run_id} eps={eps} ARI={run_metrics[0]:.4f} MSE={mse_run:.4f} time={runtime_run:.1f}s rr_p={rr_p_run}")

        # aggregate DPMS
        for key, lst in zip(["ARI","NMI","Silhouette","MSE"], [ari_list, nmi_list, sil_list, mse_list]):
            if len(lst) > 0:
                dpms_metrics[key].append(np.mean(lst))
                dpms_err[key].append(np.std(lst))
            else:
                dpms_metrics[key].append(np.nan)
                dpms_err[key].append(np.nan)
        dpms_times_mean.append(float(np.mean(runtimes_list)) if runtimes_list else np.nan)
        dpms_times_std.append(float(np.std(runtimes_list, ddof=1)) if len(runtimes_list) > 1 else 0.0)

        # DP-KMeans (parallel)
        ari_list, nmi_list, sil_list, mse_list, runtimes_list = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(dpkm_single_run, run, X, y_true, eps, merge_n_clusters, base_rng_seed): run for run in range(n_runs)}
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    modes_run, labels_run, run_metrics, runtime_run, seed_run = future.result()
                except Exception as e:
                    print(f"[{now_str()}] [ERROR] DP-KMeans future for run {run_id} failed: {e}")
                    continue

                modes_orig = pca_50.inverse_transform(modes_run)
                modes_scaled = scaler.transform(modes_orig)
                mse_run = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

                ari_list.append(run_metrics[0])
                nmi_list.append(run_metrics[1])
                sil_list.append(run_metrics[2])
                mse_list.append(mse_run)
                runtimes_list.append(runtime_run)

                # write per-run CSV row
                with open(dpkm_perrun_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "DP-KMeans", eps, run_id, seed_run,
                        float(run_metrics[0]), float(run_metrics[1]), float(run_metrics[2]),
                        float(mse_run), float(runtime_run), now_str()
                    ])

                print(f"[{now_str()}] [DP-KMeans main] wrote run={run_id} eps={eps} ARI={run_metrics[0]:.4f} MSE={mse_run:.4f} time={runtime_run:.2f}s")

        # aggregate DP-KMeans
        for key, lst in zip(["ARI","NMI","Silhouette","MSE"], [ari_list, nmi_list, sil_list, mse_list]):
            if len(lst) > 0:
                dpkm_metrics[key].append(np.mean(lst))
                dpkm_err[key].append(np.std(lst))
            else:
                dpkm_metrics[key].append(np.nan)
                dpkm_err[key].append(np.nan)
        dpkm_times_mean.append(float(np.mean(runtimes_list)) if runtimes_list else np.nan)
        dpkm_times_std.append(float(np.std(runtimes_list, ddof=1)) if len(runtimes_list) > 1 else 0.0)

    # ---------------------------
    # Save aggregated privacy-utility CSV (one row per eps & algorithm)
    # ---------------------------
    agg_csv = os.path.join(results_dir, "privacy_utility_metrics_aggregated.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Eps_modes", "Algorithm",
            "ARI", "NMI", "Silhouette", "MSE",
            "Std_ARI", "Std_NMI", "Std_Silhouette", "Std_MSE",
            "Mean_Runtime(s)", "Std_Runtime(s)"
        ])
        for i, eps in enumerate(eps_modes_list):
            writer.writerow([
                eps, "DP-GRAMS-C",
                dpms_metrics["ARI"][i], dpms_metrics["NMI"][i], dpms_metrics["Silhouette"][i], dpms_metrics["MSE"][i],
                dpms_err["ARI"][i], dpms_err["NMI"][i], dpms_err["Silhouette"][i], dpms_err["MSE"][i],
                dpms_times_mean[i], dpms_times_std[i]
            ])
            writer.writerow([
                eps, "DP-KMeans",
                dpkm_metrics["ARI"][i], dpkm_metrics["NMI"][i], dpkm_metrics["Silhouette"][i], dpkm_metrics["MSE"][i],
                dpkm_err["ARI"][i], dpkm_err["NMI"][i], dpkm_err["Silhouette"][i], dpkm_err["MSE"][i],
                dpkm_times_mean[i], dpkm_times_std[i]
            ])
    print(f"[{now_str()}] Aggregated privacy-utility metrics saved to: {agg_csv}")

    # ---------------------------
    # Privacy-utility plot
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_names = ["ARI","NMI","Silhouette","MSE"]
    for ax, metric in zip(axes.flat, metrics_names):
        ax.errorbar(eps_modes_list, dpms_metrics[metric], yerr=dpms_err[metric], marker='o', linestyle='-', label="DP-GRAMS-C")
        ax.errorbar(eps_modes_list, dpkm_metrics[metric], yerr=dpkm_err[metric], marker='s', linestyle='-', label="DP-KMeans")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\epsilon_{\mathrm{modes}}$")
        ax.set_ylabel(metric)
        ax.set_title(f"Privacy-Utility: {metric}")
        ax.grid(True, linestyle="--", alpha=0.6)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Privacy-Utility Tradeoff for MNIST Dataset", fontsize=16, fontweight="bold", y=0.98)

    save_path_privacy = os.path.join(results_dir, "mnist_privacy_utility_standardized.png")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path_privacy, dpi=300)
    plt.show()
    print(f"[{now_str()}] Privacy-utility tradeoff saved to: {save_path_privacy}")
    print(f"[{now_str()}] All MNIST experiments completed. Results saved in: {results_dir}")

# ---------------------------
# Entry point (safe for macOS/Windows)
# ---------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

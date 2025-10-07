# iris_compare_centroids_wo_k.py
import numpy as np
import os
import sys
import time
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dpms_cluster import dpms_private_rr
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes


results_dir = "results/iris_centroid_comparison_wo_k"
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

# ---------------------------
# Load data
# ---------------------------
iris = load_iris()
X_raw = iris.data
y_true = iris.target

n, d = X_raw.shape

# ---------------------------
# Parameters
# ---------------------------
merge_n_clusters = 3  # Iris has 3 classes
ms_T = int(np.log(n))
ms_p = 1
eps_rr_high = 1e2
delta = 1e-5
p_seed = 1
base_rng_seed = 12
n_runs = 20

# ---------------------------
# Standardized version for MSE only
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in np.unique(y_true)])

# ---------------------------
# Use raw X for clustering
# ---------------------------
X = X_raw
true_means = np.array([X[y_true == c].mean(axis=0) for c in np.unique(y_true)])

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


# ---------------------------
# 1) Mean-Shift 
# ---------------------------
rng_ms = np.random.default_rng(base_rng_seed)
raw_ms_modes, ms_time = timed(mean_shift, X, None, ms_T, None, ms_p, rng_ms)
ms_merged_modes = merge_modes(raw_ms_modes)
dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1)
labels_ms = relabel_clusters(y_true, labels_ms)

# ---------------------------
# 2) DPMS-C with high eps_rr
# ---------------------------
rng_dpms = np.random.default_rng(base_rng_seed)
(modes_dpms, labels_dpms, rr_p), dpms_time = timed(
    dpms_private_rr,
    data=X,
    epsilon_modes=1,
    epsilon_rr=eps_rr_high,
    delta=delta,
    p_seed=p_seed,
    rng=rng_dpms,
    # k_est=merge_n_clusters,
    bandwidth_multiplier = 4
)
labels_dpms = relabel_clusters(y_true, labels_dpms)

# ---------------------------
# 3) Standard KMeans
# ---------------------------
kmeans = KMeans(n_clusters=merge_n_clusters, random_state=base_rng_seed, n_init=10)
labels_km, km_time = timed(kmeans.fit_predict, X)
modes_km = kmeans.cluster_centers_
labels_km = relabel_clusters(y_true, labels_km)

# ---------------------------
# 4) DP KMeans
# ---------------------------
from diffprivlib.models import KMeans as DPKMeans
dp_kmeans = DPKMeans(n_clusters=merge_n_clusters, epsilon=1, random_state=base_rng_seed)
try:
    labels_dpkm, dpkm_time = timed(dp_kmeans.fit, X)
except ValueError:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    labels_dpkm, dpkm_time = timed(dp_kmeans.fit, X, bounds=(mins, maxs))
labels_dpkm = dp_kmeans.labels_.astype(int) if hasattr(dp_kmeans, "labels_") else dp_kmeans.predict(X)
modes_dpkm = dp_kmeans.cluster_centers_
labels_dpkm = relabel_clusters(y_true, labels_dpkm)

# ---------------------------
# Compute metrics and MSE (MSE on standardized X)
# ---------------------------
algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
modes_list = [ms_merged_modes, modes_dpms, modes_km, modes_dpkm]
runtimes = [ms_time, dpms_time, km_time, dpkm_time]

metrics = {}
mse_centroids = {}
for alg, labels, modes in zip(algorithms, labels_list, modes_list):
    metrics[alg] = compute_metrics(y_true, labels, X)  # metrics on raw
    modes_scaled = scaler.transform(modes)             # scale modes
    mse_centroids[alg] = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())

# Save CSV and TXT results
csv_file = os.path.join(results_dir, "clustering_metrics.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "ARI", "NMI", "Silhouette", "MSE", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        writer.writerow([alg, metrics[alg][0], metrics[alg][1], metrics[alg][2], mse_centroids[alg], rt])

txt_file = os.path.join(results_dir, "clustering_metrics.txt")
with open(txt_file, "w") as f:
    for alg, rt in zip(algorithms, runtimes):
        f.write(f"{alg}: ARI={metrics[alg][0]:.4f}, NMI={metrics[alg][1]:.4f}, "
                f"Silhouette={metrics[alg][2]:.4f}, MSE={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n")

print("\nMetrics (raw features for clustering, MSE on scaled features):")
print(f"{'Alg':<12} {'ARI':>6} {'NMI':>6} {'Silhouette':>10} {'MSE_centroids':>12} {'Runtime(s)':>12}")
for alg, rt in zip(algorithms, runtimes):
    vals = metrics[alg]
    print(f"{alg:<12} {vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:10.3f} {mse_centroids[alg]:12.6f} {rt:12.4f}")

# ---------------------------
# PCA 2D visualization with one global legend (fixed)
# ---------------------------
pca = PCA(n_components=2, random_state=base_rng_seed)
X_2d = pca.fit_transform(X)
true_means_2d = pca.transform(true_means)
modes_2d_list = [pca.transform(m) for m in modes_list]

fig, axes = plt.subplots(1, 4, figsize=(20, 8))
palette = sns.color_palette("tab10", merge_n_clusters)

# Create global handles/labels only once
global_handles = []
global_labels = []

for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    for i, color in enumerate(palette):
        sc = ax.scatter(X_2d[labels_pred == i, 0], X_2d[labels_pred == i, 1],
                        c=[color], s=40, alpha=0.7)
        if len(global_handles) < merge_n_clusters:  # only once per cluster
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")
    # true means
    true_sc = ax.scatter(true_means_2d[:, 0], true_means_2d[:, 1],
                         marker="X", c="magenta", s=180, linewidths=2)
    if "True means" not in global_labels:
        global_handles.append(true_sc)
        global_labels.append("True means")
    # estimated modes
    modes_sc = ax.scatter(modes_2d[:, 0], modes_2d[:, 1],
                          marker="X", c="blue", s=120, linewidths=2)
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")

    ax.set_title(alg, fontsize=16)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# One global legend at the bottom
legend = fig.legend(global_handles, global_labels, fontsize=10,
                    loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.01),
                    title="Cluster Assignments & Centroids")
plt.setp(legend.get_title(), fontsize=12, fontweight="bold")
fig.suptitle("Clustering Comparison on Iris Dataset", fontsize=20, y=0.90)

save_path_clusters = os.path.join(results_dir, "iris_clustering_comparison.png")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(save_path_clusters, dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------
# Privacy-Utility Tradeoff vs eps_modes
# ---------------------------
eps_modes_list = [0.1, 0.5, 1, 2, 5, 10]
dpms_metrics, dpkm_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}, {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpms_err, dpkm_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}, {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpms_times_mean, dpkm_times_mean = [], []
dpms_times_std, dpkm_times_std = [], []

for eps in eps_modes_list:
    # DPMS-C
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    dpms_times = []
    for run in range(n_runs):
        rng_run = np.random.default_rng(base_rng_seed + run)
        # time the dpms run explicitly to capture runtime per run
        t0 = time.time()
        modes_dpms_run, labels_dpms_run, _ = dpms_private_rr(
            data=X, epsilon_modes=eps, epsilon_rr=eps_rr_high, delta=delta, p_seed=p_seed,
            rng=rng_run, bandwidth_multiplier=4
        )
        t_dpms = time.time() - t0
        dpms_times.append(t_dpms)

        labels_dpms_run = relabel_clusters(y_true, labels_dpms_run)
        run_metrics = compute_metrics(y_true, labels_dpms_run, X)
        modes_scaled = scaler.transform(modes_dpms_run)
        ari_list.append(run_metrics[0])
        nmi_list.append(run_metrics[1])
        sil_list.append(run_metrics[2])
        mse_list.append(mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy()))
    dpms_metrics["ARI"].append(np.mean(ari_list))
    dpms_metrics["NMI"].append(np.mean(nmi_list))
    dpms_metrics["Silhouette"].append(np.mean(sil_list))
    dpms_metrics["MSE"].append(np.mean(mse_list))
    dpms_err["ARI"].append(np.std(ari_list))
    dpms_err["NMI"].append(np.std(nmi_list))
    dpms_err["Silhouette"].append(np.std(sil_list))
    dpms_err["MSE"].append(np.std(mse_list))
    dpms_times_mean.append(float(np.mean(dpms_times)))
    dpms_times_std.append(float(np.std(dpms_times, ddof=1)))

    # DP-KMeans
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    dpkm_times = []
    for run in range(n_runs):
        t0 = time.time()
        dp_kmeans_run = DPKMeans(n_clusters=merge_n_clusters, epsilon=eps, random_state=base_rng_seed + run)
        try:
            dp_kmeans_run.fit(X)
        except ValueError:
            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            dp_kmeans_run.fit(X, bounds=(mins, maxs))
        t_dpkm = time.time() - t0
        dpkm_times.append(t_dpkm)

        labels_dpkm_run = dp_kmeans_run.labels_.astype(int) if hasattr(dp_kmeans_run, "labels_") else dp_kmeans_run.predict(X)
        labels_dpkm_run = relabel_clusters(y_true, labels_dpkm_run)
        modes_dpkm_run = dp_kmeans_run.cluster_centers_
        run_metrics = compute_metrics(y_true, labels_dpkm_run, X)
        modes_scaled = scaler.transform(modes_dpkm_run)
        ari_list.append(run_metrics[0])
        nmi_list.append(run_metrics[1])
        sil_list.append(run_metrics[2])
        mse_list.append(mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy()))
    dpkm_metrics["ARI"].append(np.mean(ari_list))
    dpkm_metrics["NMI"].append(np.mean(nmi_list))
    dpkm_metrics["Silhouette"].append(np.mean(sil_list))
    dpkm_metrics["MSE"].append(np.mean(mse_list))
    dpkm_err["ARI"].append(np.std(ari_list))
    dpkm_err["NMI"].append(np.std(nmi_list))
    dpkm_err["Silhouette"].append(np.std(sil_list))
    dpkm_err["MSE"].append(np.std(mse_list))
    dpkm_times_mean.append(float(np.mean(dpkm_times)))
    dpkm_times_std.append(float(np.std(dpkm_times, ddof=1)))

# Save privacy-utility CSV (include runtime means and std)
csv_file = os.path.join(results_dir, "privacy_utility_metrics.csv")
with open(csv_file, "w", newline="") as f:
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

print("All Iris experiments completed. Results saved in:", results_dir)

# ---------------------------
# Plot Privacy-Utility Tradeoff
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics_names = ["ARI","NMI","Silhouette","MSE"]

for ax, metric in zip(axes.flat, metrics_names):
    ax.errorbar(eps_modes_list, dpms_metrics[metric], yerr=dpms_err[metric], marker='o', label="DP-GRAMS-C")
    ax.errorbar(eps_modes_list, dpkm_metrics[metric], yerr=dpkm_err[metric], marker='s', label="DP-KMeans")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon_{\mathrm{modes}}$")
    ax.set_ylabel(metric)
    ax.set_title(f"Privacy-Utility: {metric}")
    ax.grid(True, linestyle="--", alpha=0.6)

handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
fig.suptitle("Privacy-Utility Tradeoff for Iris Dataset", fontsize=16, fontweight="bold", y=0.98)

save_path_privacy = os.path.join(results_dir, "iris_privacy_utility_standardized.png")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(save_path_privacy, dpi=300)
plt.show()

print("Privacy-utility tradeoff saved to:", save_path_privacy)

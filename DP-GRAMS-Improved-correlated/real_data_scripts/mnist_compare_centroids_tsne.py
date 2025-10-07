# mnist_compare_centroids_tsne.py
import numpy as np
import os
import sys
import time
import csv
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dpms_cluster import dpms_private_rr
from main_scripts.mode_matching_mse import mode_matching_mse
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative

# DP KMeans
from diffprivlib.models import KMeans as DPKMeans

# ---------------------------
# Parameters (tweak as needed)
# ---------------------------
merge_n_clusters = 10          # MNIST has 10 classes
ms_T = 10
ms_p = 0.1
eps_rr_high = 1e2
delta = 1e-5
p_seed = 0.1
base_rng_seed = 41
n_runs = 20                   # privacy-utility repeats (reduced for speed)
n_subsample = 10000            # subsample size for embedding & clustering
embedding_dim = 10             # requested embedding dimension (t-SNE or PCA fallback)
results_dir = os.path.abspath("results/mnist_centroid_comparison_tsne_10d")
os.makedirs(results_dir, exist_ok=True)

sns.set(style="whitegrid", context="talk")

# ---------------------------
# Load & subsample MNIST
# ---------------------------
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_all = mnist.data.astype(np.float32)
y_all = mnist.target.astype(int)
print("Original MNIST shape:", X_all.shape)

rng = np.random.default_rng(base_rng_seed)
if n_subsample is None or n_subsample >= X_all.shape[0]:
    X_raw = X_all.copy()
    y_true = y_all.copy()
else:
    indices_sub = rng.choice(X_all.shape[0], size=n_subsample, replace=False)
    X_raw = X_all[indices_sub]
    y_true = y_all[indices_sub]
print("Working MNIST shape:", X_raw.shape)

# ---------------------------
# Standardize pixel space for MSE
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
unique_classes = np.unique(y_true)
true_means_scaled = np.array([X_scaled[y_true == c].mean(axis=0) for c in unique_classes])

# ---------------------------
# t-SNE (with caching) and robust fallback to PCA
# ---------------------------
tsne_cache = os.path.join(results_dir, f"mnist_tsne_{embedding_dim}d.npy")
used_embedding = None
start_tsne = time.time()
try:
    if os.path.exists(tsne_cache):
        print(f"Loading cached embedding from {tsne_cache} ...")
        X_emb = np.load(tsne_cache)
        used_embedding = "cached"
    else:
        print(f"Computing embedding (attempting t-SNE {embedding_dim}D). This can take a while...")
        # For embedding_dim >= 4 we must use method='exact' (barnes_hut restricted).
        method = "exact" if embedding_dim >= 4 else "barnes_hut"
        tsne = TSNE(n_components=embedding_dim, random_state=base_rng_seed, init="pca",
                    learning_rate="auto", method=method)
        X_emb = tsne.fit_transform(X_raw)
        np.save(tsne_cache, X_emb)
        used_embedding = "tsne"
    tsne_time = time.time() - start_tsne
    print(f"Embedding runtime: {tsne_time:.2f}s (method={used_embedding})")
except Exception as e:
    # If TSNE fails (OOM, killed, or other), fallback to PCA embedding of same dim
    print("t-SNE failed or was too heavy — falling back to PCA embedding.")
    print("t-SNE exception:", repr(e))
    start_pca = time.time()
    pca_emb = PCA(n_components=embedding_dim, random_state=base_rng_seed)
    X_emb = pca_emb.fit_transform(X_raw)
    tsne_time = time.time() - start_pca
    used_embedding = "pca_fallback"
    # If we want to cache fallback PCA embedding:
    np.save(tsne_cache, X_emb)
    print(f"PCA fallback embedding runtime: {tsne_time:.2f}s (saved to cache)")

# For inverse mapping back to pixel space we build a PCA model with the SAME embedding_dim.
# This PCA will be used as a linear approx to invert embedding->pixels for MSE computation.
pca_fallback = PCA(n_components=embedding_dim, random_state=base_rng_seed)
pca_fallback.fit(X_raw)  # fit on pixel space to allow inverse_transform of embedding vectors

# True means in embedding space (for visualization / some embedding-based metrics)
true_means_emb = np.array([X_emb[y_true == c].mean(axis=0) for c in unique_classes])

# ---------------------------
# Helper functions
# ---------------------------
def compute_metrics(y_true_local, labels_local, X_local):
    ari = adjusted_rand_score(y_true_local, labels_local)
    nmi = normalized_mutual_info_score(y_true_local, labels_local)
    sil = silhouette_score(X_local, labels_local) if len(np.unique(labels_local)) > 1 else np.nan
    return ari, nmi, sil

def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def relabel_clusters(y_true_local, y_pred_local, n_clusters=10):
    # Build confusion matrix with rows=true classes, cols=predicted (0..k-1)
    cm = confusion_matrix(y_true_local, y_pred_local, labels=range(n_clusters))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(lbl, lbl) for lbl in y_pred_local])

def modes_to_pixel_mse(modes_emb):
    """
    Map modes from embedding_dim -> pixel space using pca_fallback.inverse_transform,
    then standardize with the scaler and compute mode_matching_mse vs true_means_scaled.
    modes_emb shape should be (k, embedding_dim).
    """
    if modes_emb is None or len(modes_emb) == 0:
        return np.nan
    # ensure array shape
    modes_emb = np.asarray(modes_emb)
    if modes_emb.ndim == 1:
        modes_emb = modes_emb.reshape(1, -1)
    # inverse transform using PCA fallback and scale
    try:
        modes_orig = pca_fallback.inverse_transform(modes_emb)
        modes_scaled = scaler.transform(modes_orig)
        mse = mode_matching_mse(true_means_scaled.copy(), modes_scaled.copy())
        return mse
    except Exception as e:
        print("Error mapping modes to pixel space for MSE:", e)
        return np.nan

# ---------------------------
# 1) Mean-Shift + agglomerative merge
# ---------------------------
rng_ms = np.random.default_rng(base_rng_seed)
raw_ms_modes, ms_time = timed(mean_shift, X_emb, None, ms_T, None, ms_p, rng_ms)
# merge into desired number of clusters
ms_merged_modes = merge_modes_agglomerative(raw_ms_modes, n_clusters=merge_n_clusters, random_state=base_rng_seed)
# assign labels by closest merged mode
dists_ms = np.linalg.norm(X_emb[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1)
labels_ms = relabel_clusters(y_true, labels_ms, merge_n_clusters)

# ---------------------------
# 2) DPMS-C with high eps_rr (single run for baseline)
# ---------------------------
rng_dpms = np.random.default_rng(base_rng_seed)
(dpms_modes_baseline, labels_dpms_baseline, rr_p), dpms_time = timed(
    dpms_private_rr,
    data=X_emb,
    epsilon_modes=1,
    epsilon_rr=eps_rr_high,
    delta=delta,
    data_bound_R=np.max(np.linalg.norm(X_emb, axis=1)),
    p_seed=p_seed,
    rng=rng_dpms,
    k_est=merge_n_clusters,
    clip_multiplier=0.1
)
labels_dpms = relabel_clusters(y_true, labels_dpms_baseline, merge_n_clusters)

# ---------------------------
# 3) Standard KMeans
# ---------------------------
kmeans = KMeans(n_clusters=merge_n_clusters, random_state=base_rng_seed, n_init=10)
(labels_km, km_time) = timed(kmeans.fit_predict, X_emb)
modes_km = kmeans.cluster_centers_
labels_km = relabel_clusters(y_true, labels_km, merge_n_clusters)

# ---------------------------
# 4) DP-KMeans (single baseline run)
# ---------------------------
dp_kmeans = DPKMeans(n_clusters=merge_n_clusters, epsilon=1, random_state=base_rng_seed)
try:
    (labels_dpkm, dpkm_time) = timed(dp_kmeans.fit, X_emb)
except ValueError:
    mins_emb = X_emb.min(axis=0)
    maxs_emb = X_emb.max(axis=0)
    (labels_dpkm, dpkm_time) = timed(dp_kmeans.fit, X_emb, bounds=(mins_emb, maxs_emb))
labels_dpkm = dp_kmeans.labels_.astype(int) if hasattr(dp_kmeans, "labels_") else dp_kmeans.predict(X_emb)
modes_dpkm = dp_kmeans.cluster_centers_
labels_dpkm = relabel_clusters(y_true, labels_dpkm, merge_n_clusters)

# ---------------------------
# Compute baseline metrics and centroid MSEs (pixel-space)
# ---------------------------
algorithms = ["MS Clustering", "DP-GRAMS-C", "KMeans", "DP-KMeans"]
labels_list = [labels_ms, labels_dpms, labels_km, labels_dpkm]
modes_list = [ms_merged_modes, dpms_modes_baseline, modes_km, modes_dpkm]
runtimes = [ms_time, dpms_time, km_time, dpkm_time]

metrics = {}
mse_centroids = {}
for alg, labels, modes, rt in zip(algorithms, labels_list, modes_list, runtimes):
    metrics[alg] = compute_metrics(y_true, labels, X_emb)
    mse_centroids[alg] = modes_to_pixel_mse(modes)

# ---------------------------
# Save clustering metrics (baseline) CSV/TXT
# ---------------------------
csv_file = os.path.join(results_dir, "clustering_metrics_tsne_10d.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "ARI", "NMI", "Silhouette", "MSE_pixels", "Runtime(s)"])
    for alg, rt in zip(algorithms, runtimes):
        writer.writerow([alg, metrics[alg][0], metrics[alg][1], metrics[alg][2], mse_centroids[alg], rt])

txt_file = os.path.join(results_dir, "clustering_metrics_tsne_10d.txt")
with open(txt_file, "w") as f:
    for alg, rt in zip(algorithms, runtimes):
        vals = metrics[alg]
        f.write(f"{alg}: ARI={vals[0]:.4f}, NMI={vals[1]:.4f}, Silhouette={vals[2]:.4f}, "
                f"MSE_pixels={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s\n")

print("\nBaseline metrics (embedding metrics; MSE on standardized pixels):")
for alg, rt in zip(algorithms, runtimes):
    vals = metrics[alg]
    print(f"{alg:<12} ARI={vals[0]:.3f}, NMI={vals[1]:.3f}, Silhouette={vals[2]:.3f}, "
          f"MSE_pixels={mse_centroids[alg]:.6f}, Runtime={rt:.4f}s")

# ---------------------------
# PCA 2D visualization for plotting clarity
# ---------------------------
pca_2d = PCA(n_components=2, random_state=base_rng_seed)
X_2d = pca_2d.fit_transform(X_emb)
true_means_2d = pca_2d.transform(true_means_emb)
modes_2d_list = [pca_2d.transform(m) for m in modes_list]

fig, axes = plt.subplots(1, 4, figsize=(20, 8))
palette = sns.color_palette("tab10", merge_n_clusters)
global_handles, global_labels = [], []

for ax, alg, labels_pred, modes_2d in zip(axes, algorithms, labels_list, modes_2d_list):
    for i, color in enumerate(palette):
        sc = ax.scatter(X_2d[labels_pred == i, 0], X_2d[labels_pred == i, 1], c=[color], s=35, alpha=0.7)
        if len(global_handles) < merge_n_clusters:
            global_handles.append(sc)
            global_labels.append(f"Cluster {i}")
    true_sc = ax.scatter(true_means_2d[:, 0], true_means_2d[:, 1], marker="X", c="magenta", s=140, linewidths=2)
    if "True means" not in global_labels:
        global_handles.append(true_sc)
        global_labels.append("True means")
    modes_sc = ax.scatter(modes_2d[:, 0], modes_2d[:, 1], marker="X", c="blue", s=100, linewidths=2)
    if "Estimated modes" not in global_labels:
        global_handles.append(modes_sc)
        global_labels.append("Estimated modes")
    ax.set_title(alg, fontsize=14)
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")

legend = fig.legend(global_handles, global_labels, fontsize=10, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.93))
fig.suptitle(f"Clustering Comparison on MNIST (embedding method={used_embedding}, embdim={embedding_dim})", fontsize=16, y=0.98)
save_path_clusters = os.path.join(results_dir, "mnist_clustering_comparison_tsne_10d.png")
plt.tight_layout(rect=[0,0,1,0.92])
plt.savefig(save_path_clusters, dpi=300)
plt.show()

# ---------------------------
# Privacy-Utility Tradeoff: vary eps_modes, repeated runs, record runtimes
# ---------------------------
eps_modes_list = [0.1, 0.5, 2, 5, 10]

dpms_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpkm_metrics = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpms_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpkm_err = {m: [] for m in ["ARI","NMI","Silhouette","MSE"]}
dpms_times_mean, dpms_times_std = [], []
dpkm_times_mean, dpkm_times_std = [], []

for eps in eps_modes_list:
    # DPMS runs
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    times_list = []
    for run in range(n_runs):
        t0 = time.time()
        rng_run = np.random.default_rng(base_rng_seed + run)
        modes_dpms_run, labels_dpms_run, _ = dpms_private_rr(
            data=X_emb, epsilon_modes=eps, epsilon_rr=eps_rr_high, delta=delta,
            data_bound_R=np.max(np.linalg.norm(X_emb, axis=1)), p_seed=p_seed,
            rng=rng_run, k_est=merge_n_clusters, clip_multiplier=0.1
        )
        dt = time.time() - t0
        times_list.append(dt)

        labels_run = relabel_clusters(y_true, labels_dpms_run, merge_n_clusters)
        ari, nmi, sil = compute_metrics(y_true, labels_run, X_emb)
        ari_list.append(ari); nmi_list.append(nmi); sil_list.append(sil)
        mse_list.append(modes_to_pixel_mse(modes_dpms_run))

    dpms_metrics["ARI"].append(np.mean(ari_list))
    dpms_metrics["NMI"].append(np.mean(nmi_list))
    dpms_metrics["Silhouette"].append(np.mean(sil_list))
    dpms_metrics["MSE"].append(np.mean(mse_list))
    dpms_err["ARI"].append(np.std(ari_list))
    dpms_err["NMI"].append(np.std(nmi_list))
    dpms_err["Silhouette"].append(np.std(sil_list))
    dpms_err["MSE"].append(np.std(mse_list))
    dpms_times_mean.append(float(np.mean(times_list)))
    dpms_times_std.append(float(np.std(times_list, ddof=1)))

    # DP-KMeans runs
    ari_list, nmi_list, sil_list, mse_list = [], [], [], []
    times_list = []
    for run in range(n_runs):
        t0 = time.time()
        dp_kmeans_run = DPKMeans(n_clusters=merge_n_clusters, epsilon=eps, random_state=base_rng_seed + run)
        try:
            dp_kmeans_run.fit(X_emb)
        except ValueError:
            mins_emb = X_emb.min(axis=0)
            maxs_emb = X_emb.max(axis=0)
            dp_kmeans_run.fit(X_emb, bounds=(mins_emb, maxs_emb))
        dt = time.time() - t0
        times_list.append(dt)

        labels_run = dp_kmeans_run.labels_.astype(int) if hasattr(dp_kmeans_run, "labels_") else dp_kmeans_run.predict(X_emb)
        labels_run = relabel_clusters(y_true, labels_run, merge_n_clusters)
        ari, nmi, sil = compute_metrics(y_true, labels_run, X_emb)
        ari_list.append(ari); nmi_list.append(nmi); sil_list.append(sil)
        mse_list.append(modes_to_pixel_mse(dp_kmeans_run.cluster_centers_))

    dpkm_metrics["ARI"].append(np.mean(ari_list))
    dpkm_metrics["NMI"].append(np.mean(nmi_list))
    dpkm_metrics["Silhouette"].append(np.mean(sil_list))
    dpkm_metrics["MSE"].append(np.mean(mse_list))
    dpkm_err["ARI"].append(np.std(ari_list))
    dpkm_err["NMI"].append(np.std(nmi_list))
    dpkm_err["Silhouette"].append(np.std(sil_list))
    dpkm_err["MSE"].append(np.std(mse_list))
    dpkm_times_mean.append(float(np.mean(times_list)))
    dpkm_times_std.append(float(np.std(times_list, ddof=1)))

# Save privacy-utility CSV
csv_file = os.path.join(results_dir, "privacy_utility_metrics_tsne_10d.csv")
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

# Plot Privacy-Utility Tradeoff
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
fig.suptitle(f"Privacy-Utility Tradeoff for MNIST (emb={used_embedding}, embdim={embedding_dim})", fontsize=16, fontweight="bold", y=0.98)
save_path_privacy = os.path.join(results_dir, "mnist_privacy_utility_tsne_10d.png")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(save_path_privacy, dpi=300)
plt.show()

print(f"All experiments completed. Results saved in: {results_dir}")

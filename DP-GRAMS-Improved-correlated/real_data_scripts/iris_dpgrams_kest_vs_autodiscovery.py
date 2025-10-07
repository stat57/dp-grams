# iris_dpgrams_kest_vs_autodiscovery.py
"""
Compare Mean-Shift to two DP-GRAMS-C variants:
  - DP-GRAMS-C with k_est provided (k_est = 3)
  - DP-GRAMS-C without k_est (adaptive merging, more like MS)
Both use eps_rr fixed (eps_rr = 3, per your iris.py).
No centroid MSE here; report ARI/NMI/Silhouette and privacy-utility tradeoff.
"""

import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local project imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from dpms_cluster import dpms_private_rr
from main_scripts.ms import mean_shift
from main_scripts.merge import merge_modes_agglomerative

sns.set(style="whitegrid", context="talk")
RESULTS_DIR = "results/dpgrams_kest_vs_autodiscovery"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Params
# ---------------------------
merge_n_clusters = 3
eps_rr = 3.0       
delta = 1e-5
p_seed = 0.6
n_runs = 20
rng_seed_base = 123
eps_modes_list = [0.1, 0.5, 1, 2, 5, 10]

# ---------------------------
# Helpers
# ---------------------------
def compute_metrics(y_true, labels, X):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    sil = silhouette_score(X, labels)
    return ari, nmi, sil

# ---------------------------
# Load data + MS baseline
# ---------------------------
iris = load_iris()
X = iris.data
y_true = iris.target

raw_ms_modes = mean_shift(X, initial_modes=None, T=20, bandwidth=None, p=0.6)
ms_merged_modes = merge_modes_agglomerative(raw_ms_modes, n_clusters=merge_n_clusters)
dists_ms = np.linalg.norm(X[:, None, :] - ms_merged_modes[None, :, :], axis=2)
labels_ms = np.argmin(dists_ms, axis=1)
metrics_ms = compute_metrics(y_true, labels_ms, X)
print(f"[INFO] Mean-Shift baseline metrics (ARI,NMI,Sil) = {metrics_ms}")

# ---------------------------
# Sweep eps_modes for DP-GRAMS-C with k_est known vs unknown
# ---------------------------
results = {
    "eps": [],
    "MS_ARI": [], "MS_NMI": [], "MS_SIL": [],
    "DP_KKNOWN_ARI": [], "DP_KKNOWN_NMI": [], "DP_KKNOWN_SIL": [],
    "DP_KFREE_ARI": [], "DP_KFREE_NMI": [], "DP_KFREE_SIL": []
}

for eps_modes in eps_modes_list:
    print(f"\n[EXPERIMENT] eps_modes = {eps_modes}")
    metrics_kknown_runs = []
    metrics_kfree_runs = []

    for run in range(n_runs):
        rng = np.random.default_rng(rng_seed_base + run)

        # DP-GRAMS-C with k_est known
        merged_modes_k, labels_k, rr_p_k = dpms_private_rr(
            data=X,
            epsilon_modes=eps_modes,
            epsilon_rr=eps_rr,
            delta=delta,
            data_bound_R=np.max(np.linalg.norm(X, axis=1)),
            p_seed=p_seed,
            rng=rng,
            k_est=merge_n_clusters
        )
        metrics_kknown_runs.append(compute_metrics(y_true, labels_k, X))

        # DP-GRAMS-C without k_est (adaptive merging)
        merged_modes_f, labels_f, rr_p_f = dpms_private_rr(
            data=X,
            epsilon_modes=eps_modes,
            epsilon_rr=eps_rr,
            delta=delta,
            data_bound_R=np.max(np.linalg.norm(X, axis=1)),
            p_seed=p_seed,
            rng=rng,
            k_est=None
        )
        metrics_kfree_runs.append(compute_metrics(y_true, labels_f, X))

    metrics_kknown_runs = np.array(metrics_kknown_runs)
    metrics_kfree_runs = np.array(metrics_kfree_runs)

    results["eps"].append(eps_modes)
    results["MS_ARI"].append(metrics_ms[0]); results["MS_NMI"].append(metrics_ms[1]); results["MS_SIL"].append(metrics_ms[2])
    results["DP_KKNOWN_ARI"].append(metrics_kknown_runs[:,0].mean())
    results["DP_KKNOWN_NMI"].append(metrics_kknown_runs[:,1].mean())
    results["DP_KKNOWN_SIL"].append(metrics_kknown_runs[:,2].mean())
    results["DP_KFREE_ARI"].append(metrics_kfree_runs[:,0].mean())
    results["DP_KFREE_NMI"].append(metrics_kfree_runs[:,1].mean())
    results["DP_KFREE_SIL"].append(metrics_kfree_runs[:,2].mean())

# ---------------------------
# Save + Plot
# ---------------------------
np.savetxt(os.path.join(RESULTS_DIR, "dpgrams_kest_vs_free.csv"),
            np.column_stack([results["eps"],
                              results["MS_ARI"], results["MS_NMI"], results["MS_SIL"],
                              results["DP_KKNOWN_ARI"], results["DP_KKNOWN_NMI"], results["DP_KKNOWN_SIL"],
                              results["DP_KFREE_ARI"], results["DP_KFREE_NMI"], results["DP_KFREE_SIL"]]),
            header="eps,MS_ARI,MS_NMI,MS_SIL,DP_KKNOWN_ARI,DP_KKNOWN_NMI,DP_KKNOWN_SIL,DP_KFREE_ARI,DP_KFREE_NMI,DP_KFREE_SIL",
            delimiter=",")

plt.figure(figsize=(8,6))
plt.plot(results["eps"], results["MS_ARI"], marker='o', label="MS (baseline)")
plt.plot(results["eps"], results["DP_KKNOWN_ARI"], marker='s', label="DP-GRAMS-C (k_est known)")
plt.plot(results["eps"], results["DP_KFREE_ARI"], marker='^', label="DP-GRAMS-C (k_est unknown)")
plt.xscale("log")
plt.xlabel(r"$\epsilon_{\mathrm{modes}}$")
plt.ylabel("ARI")
plt.title("ARI: MS vs DP-GRAMS-C (k_est known vs unknown) [eps_rr fixed]")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ari_dpgrams_kest_vs_free.png"), dpi=200)
plt.close()

print(f"[DONE] Results saved to {RESULTS_DIR}")

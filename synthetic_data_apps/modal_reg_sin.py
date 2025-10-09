import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os, sys
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthetic_data_apps.dp_pms import dp_pms_simple
from PMS import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse

sns.set_context("talk")
sns.set_style("white")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11})

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

sigma_noise = 0.15

def generate_sin_data(n, sigma=sigma_noise, seed=None):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    X1 = rng.uniform(0,1,n1)
    X2 = rng.uniform(0,1,n2)
    X = np.concatenate([X1,X2])
    Y1 = rng.normal(1.5 + 0.5*np.sin(3*np.pi*X1), sigma, n1)
    Y2 = rng.normal(0.5*np.sin(3*np.pi*X2), sigma, n2)
    Y = np.concatenate([Y1,Y2])
    return X,Y

results_dir = "results/modal_regression_sin"
ensure_dir(results_dir)

n_values = [100,500,1000,2000]
eps_values = [0.1,0.2,0.5,1.0]
n_runs = 20
MAX_WORKERS = min(8, os.cpu_count() or 8)

perrun_csv = os.path.join(results_dir, "dp_per_run_results.csv")
if not os.path.exists(perrun_csv):
    with open(perrun_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples","epsilon","run_idx","seed","dp_mse","dp_runtime_s","timestamp"])

def dp_pms_worker(X,Y,epsilon,delta,seed,T_iter=None,verbose=False):
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    n = len(X)
    T_use = T_iter if T_iter is not None else int(np.ceil(np.log(max(2,n))))
    Y_dp = dp_pms_simple(X,Y,mesh_points=Y.copy(),epsilon=float(epsilon),
                         delta=float(delta),T=int(T_use),rng=rng,verbose=verbose)
    runtime = time.perf_counter() - t0
    return int(seed), float(runtime), Y_dp, X.copy()

def main():
    Xb,Yb = generate_sin_data(500,sigma_noise,seed=42)
    Y_pms_baseline = partial_mean_shift(Xb,Yb,mesh_points=None,T=int(np.log(500)))
    Y_dp_baseline = dp_pms_simple(Xb,Yb,mesh_points=Yb.copy(),epsilon=1,delta=1e-5,
                                  T=int(np.log(500)),rng=np.random.default_rng(42),verbose=False)
    Y_lowess = lowess(Yb,Xb,frac=0.2,return_sorted=False)

    save_text(os.path.join(results_dir,"stats_baseline.txt"),
              f"Baseline (n=500)\nPMS runtime: N/A\nDP-PMS runtime: N/A\n")

    fig,axes = plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    axes[0].scatter(Xb,Yb,color='skyblue',alpha=0.5,s=15,label='Data')
    axes[0].scatter(Xb,Y_pms_baseline,color='#1f77b4',s=25,marker='D',label='PMS')
    axes[0].plot(np.sort(Xb),Y_lowess[np.argsort(Xb)],color='orange',linewidth=2,label='LOWESS')
    axes[0].set_title('PMS vs LOWESS'); axes[0].legend()
    axes[1].scatter(Xb,Yb,color='skyblue',alpha=0.5,s=15,label='Data')
    axes[1].scatter(Xb,Y_dp_baseline,color='#d95f02',s=35,marker='X',label='DP-PMS (ε=1)')
    axes[1].set_title('DP-PMS'); axes[1].legend()
    for ax in axes:
        ax.grid(True,linestyle="--",alpha=0.4); ax.set_xlim(0,1)
    plt.suptitle('Modal Regression in Sinusoidal 2-Mixture Data',fontsize=16,y=0.97)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,"baseline_comparison.pdf"),dpi=300,bbox_inches="tight")
    plt.close()

    results=[]
    stats_summary="=== Privacy–Utility Experiments ===\n"

    for n in n_values:
        X_data,Y_data = generate_sin_data(n,sigma_noise,seed=123)
        Y_pms_est = partial_mean_shift(X_data,Y_data,mesh_points=None,T=int(np.log(n)))
        pms_mse = mode_matching_mse(np.column_stack([X_data,Y_pms_est]),
                                    np.column_stack([X_data,Y_pms_est]))

        for eps in eps_values:
            seeds = [1000*n + 10*int(eps*10) + run for run in range(n_runs)]
            dp_mses, dp_times = [], []

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(dp_pms_worker,X_data,Y_data,float(eps),1e-5,seed): (idx,seed)
                           for idx,seed in enumerate(seeds)}
                for fut in as_completed(futures):
                    run_idx,seed = futures[fut]
                    try:
                        seed_ret,runtime_ret,Y_dp_out,mesh_ret=fut.result()
                    except Exception:
                        continue
                    est_modes = np.column_stack([mesh_ret,Y_dp_out])
                    dp_mse_val = mode_matching_mse(np.column_stack([X_data,Y_pms_est]), est_modes)
                    dp_mses.append(dp_mse_val)
                    dp_times.append(runtime_ret)
                    with open(perrun_csv,"a",newline="") as f:
                        writer=csv.writer(f)
                        writer.writerow([n,eps,run_idx,seed_ret,dp_mse_val,runtime_ret,now_str()])

            if len(dp_mses)==0:
                continue
            dp_mean,dp_se=float(np.mean(dp_mses)),float(np.std(dp_mses,ddof=1)/np.sqrt(len(dp_mses)))
            dp_time_mean,dp_time_std=float(np.mean(dp_times)),float(np.std(dp_times,ddof=1) if len(dp_times)>1 else 0.0)
            results.append([n,eps,dp_mean,dp_se,pms_mse,dp_time_mean,dp_time_std])
            stats_summary+=f"n={n}, eps={eps} | DP-MSE={dp_mean:.4f}±{dp_se:.4f}, PMS-MSE={pms_mse:.4f}, DP-time={dp_time_mean:.2f}±{dp_time_std:.2f}s\n"

    csv_path=os.path.join(results_dir,"privacy_utility_results_sin_simplified.csv")
    with open(csv_path,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["n_samples","epsilon","DP_mse_mean","DP_mse_se","PMS_mse","DP_runtime_mean","DP_runtime_std"])
        writer.writerows(results)
    save_text(os.path.join(results_dir,"stats_summary.txt"),stats_summary)

    if len(results)>0:
        plt.figure(figsize=(8,6))
        palette=sns.color_palette("Set2",len(n_values))
        for i,n in enumerate(n_values):
            eps_subset=[r[1] for r in results if r[0]==n]
            mse_subset=[r[2] for r in results if r[0]==n]
            mse_errs=[r[3] for r in results if r[0]==n]
            pms_mse_val=next((r[4] for r in results if r[0]==n),np.nan)
            if len(eps_subset)>0:
                plt.errorbar(eps_subset,mse_subset,yerr=mse_errs,marker='o',linestyle='-',linewidth=2,
                             markersize=7,capsize=3,label=f"DP-PMS (n={n})",color=palette[i])
                if not np.isnan(pms_mse_val):
                    plt.hlines(pms_mse_val,min(eps_subset),max(eps_subset),colors=palette[i],
                               linestyles='dashed',linewidth=1.5,label=f"PMS baseline (n={n})")
        plt.xlabel("Privacy budget ε")
        plt.xscale('log')
        plt.ylabel("MSE")
        plt.title("Privacy–Utility Tradeoff in Sinusoidal 2-Mixture Data")
        plt.grid(axis='y',linestyle="--",alpha=0.4)
        plt.legend(ncol=1,loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir,"privacy_utility_modal_sin_simplified.pdf"),dpi=300,bbox_inches="tight")
        plt.close()

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

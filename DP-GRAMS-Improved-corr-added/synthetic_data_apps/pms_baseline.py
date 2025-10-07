import numpy as np
import os, csv, datetime, time
from PMS import partial_mean_shift
from main_scripts.mode_matching_mse import mode_matching_mse

# output path
results_dir = "results/modal_regression"
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "pms_results.csv")

# same modal data generator from your main script
true_modes_list = [3, 2, 1]
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

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# which n values to compute
n_values = [500, 1000]
results = []

for n in n_values:
    print(f"[{now_str()}] Computing PMS baseline for n={n}")
    X, Y = generate_modal_data(n, sd=0.2, seed=123)
    t0 = time.perf_counter()
    Y_pms = partial_mean_shift(X, Y, T=10)
    runtime = time.perf_counter() - t0
    mse = mode_matching_mse(np.column_stack([X, Y_pms]),
                            np.column_stack([X, Y_pms]))  # self-MSE = 0
    results.append([n, mse, runtime, now_str()])
    print(f"  -> PMS MSE={mse:.4f}, runtime={runtime:.2f}s")

# save
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_samples", "PMS_mse", "PMS_runtime", "timestamp"])
    writer.writerows(results)

print(f"[{now_str()}] Saved PMS baselines to {csv_path}")

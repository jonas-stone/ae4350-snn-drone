"""train k reproducible seeds and report mean +/- std."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotstyle; plotstyle.apply()
from train import train

# ---- config (edit these) ----
NETWORK = "SNN"          # "MLP" (~14 min/seed) or "SNN" (~2 hr/seed)
SEEDS = [0, 1, 2] #, 3, 4]  # how many independent runs
EPISODES = 6000
# ------------------------------

tag = f"{NETWORK.lower()}_{len(SEEDS)}seeds"
all_rates = []           # one success-curve per seed
episodes = None

for s in SEEDS:
    print(f"\n===== {NETWORK} seed {s} =====")
    _, _, success_history = train(train_which=NETWORK, num_episodes=EPISODES, seed=s)
    eps   = [ep for ep, _ in success_history]
    rates = [r for _, r in success_history]
    episodes = eps
    all_rates.append(rates)

R = np.array(all_rates)              # (k seeds, n checkpoints)
mean = R.mean(axis=0)
std  = R.std(axis=0)

# converged estimate per seed = mean of its last 25% of checkpoints
k_last = max(1, R.shape[1] // 4)
converged = R[:, -k_last:].mean(axis=1)     # (k,)
print(f"\n=== {NETWORK}: converged success over {len(SEEDS)} seeds ===")
print(f"  per seed: {[f'{c:.0%}' for c in converged]}")
print(f"  mean +/- std: {converged.mean():.1%} +/- {converged.std():.1%}")

# ---- figure: mean curve + std band + faint per-seed curves ----
plt.figure(figsize=(8, 5))
for rates in all_rates:
    plt.plot(episodes, [r * 100 for r in rates], color="gray", alpha=0.3, lw=1)
plt.plot(episodes, mean * 100, color="tab:blue", lw=2, label="mean")
plt.fill_between(episodes, (mean - std) * 100, (mean + std) * 100,
                 color="tab:blue", alpha=0.2, label="+/- 1 std")
plt.xlabel("episode"); plt.ylabel("success rate (%)  @ T=0.3")
plt.title(f"{NETWORK}: success over {len(SEEDS)} seeds (mean +/- std)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"figures/multiseed_{tag}.png", dpi=120); plt.close()

# ---- save aggregated data ----
with open(f"data/multiseed_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "mean_success", "std_success"] + [f"seed{s}" for s in SEEDS])
    for i, ep in enumerate(episodes):
        w.writerow([ep, f"{mean[i]:.4f}", f"{std[i]:.4f}"] + [f"{R[j, i]:.4f}" for j in range(len(SEEDS))])

print(f"saved multiseed_{tag}.png and multiseed_{tag}.csv")

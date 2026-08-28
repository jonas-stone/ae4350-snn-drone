"""firing rate vs lif threshold at initialisation, dumped to csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import torch
import numpy as np
from networks import SNNPolicy
from drone_env import DroneNavEnv

env = DroneNavEnv()
obs_list = np.array([env.reset(seed=s)[0] for s in range(40)])   # 40 diverse start observations


@torch.no_grad()
def mean_firing(threshold):
    torch.manual_seed(0)                       # identical weights every call; only threshold differs
    p = SNNPolicy(threshold=threshold); p.eval()
    n = p.fc1.out_features + p.fc2.out_features
    tot = 0.0
    for obs in obs_list:
        x = torch.as_tensor(obs, dtype=torch.float32)
        m1 = p.lif1.init_leaky(); m2 = p.lif2.init_leaky()
        for _ in range(p.C):
            spk1, m1 = p.lif1(p.fc1(x), m1); tot += spk1.sum().item()
            spk2, m2 = p.lif2(p.fc2(spk1), m2); tot += spk2.sum().item()
    return tot / (len(obs_list) * p.C * n)


os.makedirs("data", exist_ok=True)
ths = np.round(np.linspace(0.1, 1.0, 10), 2)
with open("data/threshold_sweep.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["threshold", "mean_firing_rate"])
    for t in ths:
        w.writerow([float(t), mean_firing(float(t))])
print("saved data/threshold_sweep.csv")

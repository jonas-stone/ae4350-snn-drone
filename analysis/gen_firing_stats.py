"""dump each hidden neuron average firing rate to csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import torch
import numpy as np
from networks import SNNPolicy
from drone_env import DroneNavEnv

p = SNNPolicy(C=8)
p.load_state_dict(torch.load("snn_policy_best.pt")); p.eval()


@torch.no_grad()
def layer_rates(obs):
    x = torch.as_tensor(obs, dtype=torch.float32)
    m1 = p.lif1.init_leaky(); m2 = p.lif2.init_leaky()
    s1 = torch.zeros(p.fc1.out_features); s2 = torch.zeros(p.fc2.out_features)
    for _ in range(p.C):
        spk1, m1 = p.lif1(p.fc1(x), m1); s1 += spk1
        spk2, m2 = p.lif2(p.fc2(spk1), m2); s2 += spk2
    return (s1 / p.C).numpy(), (s2 / p.C).numpy()


env = DroneNavEnv()
R1, R2 = [], []
for seed in range(40):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        r1, r2 = layer_rates(obs); R1.append(r1); R2.append(r2)
        a, _ = p.act(torch.from_numpy(obs), greedy=False, temperature=0.3)
        obs, _, term, trunc, _ = env.step(a.item()); done = term or trunc
R1, R2 = np.array(R1), np.array(R2)

os.makedirs("data", exist_ok=True)
with open("data/firing_rates.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["neuron", "layer", "avg_rate", "C"])
    for i, r in enumerate(R1.mean(axis=0)): w.writerow([i, 1, r, p.C])
    for i, r in enumerate(R2.mean(axis=0)): w.writerow([i, 2, r, p.C])
print(f"saved data/firing_rates.csv ({len(R1)} states analysed)")

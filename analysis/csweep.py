"""train a separate snn per cycle count c (expensive, run overnight)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import time
import torch
import numpy as np
from drone_env import DroneNavEnv
from networks import SNNPolicy, ValueNet
from train import run_episode, compute_returns, evaluate

C_VALUES   = [2, 4, 8, 16]
EPISODES   = 6000          # full budget (matches the main run) -> fair comparison across C
EVAL_EVERY = 300
FINAL_GOALS = 500
SEED = 0


def train_snn(C):
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = DroneNavEnv(); env.reset(seed=SEED)
    policy = SNNPolicy(C=C); critic = ValueNet()
    aopt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    best, hist, t0 = -1.0, [], time.time()
    for ep in range(EPISODES):
        lp, rw, st = run_episode(env, policy)
        ret = compute_returns(rw)
        val = critic(torch.stack(st))
        adv = ret - val.detach(); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        aloss = -(torch.stack(lp) * adv).sum(); closs = ((val - ret) ** 2).mean()
        aopt.zero_grad(); aloss.backward(); aopt.step()
        copt.zero_grad(); closs.backward(); copt.step()
        if (ep + 1) % EVAL_EVERY == 0:
            r = evaluate(policy, num_episodes=100, greedy=False, temperature=0.3)
            hist.append((ep + 1, r))
            if r > best:
                best = r; torch.save(policy.state_dict(), f"snn_C{C}_best.pt")
            print(f"  C={C} | ep {ep+1} | succ {r:.0%} | best {best:.0%} | {time.time()-t0:.0f}s")
    return hist


os.makedirs("data", exist_ok=True)
hist_rows, final_rows = [], []
for C in C_VALUES:
    print(f"=== training SNN C={C} (6000 ep) ===")
    hist = train_snn(C)
    hist_rows += [[C, ep, r] for ep, r in hist]
    best = SNNPolicy(C=C); best.load_state_dict(torch.load(f"snn_C{C}_best.pt")); best.eval()
    final = evaluate(best, num_episodes=FINAL_GOALS, greedy=False, temperature=0.3)
    final_rows.append([C, final])
    print(f"=== C={C} final ({FINAL_GOALS} goals): {final:.0%} ===\n")

with open("data/csweep_history.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["C", "episode", "success"]); w.writerows(hist_rows)
with open("data/csweep_final.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["C", "success_500"]); w.writerows(final_rows)
print("saved data/csweep_history.csv and data/csweep_final.csv")

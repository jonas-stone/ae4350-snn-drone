"""evaluate every saved model on the same 500 seeded goals."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import torch
from networks import MLPPolicy, SNNPolicy
from train import evaluate

os.makedirs("data", exist_ok=True)

N = 500          # identical seeded goals (seeds 0..499) for every model
T = 0.3          # deployment temperature

# (label, factory, checkpoint) -- SNN C-models MUST use their trained cycle count
models = [
    ("MLP (actor-critic)", lambda: MLPPolicy(),      "mlp_policy_best.pt"),
    ("SNN main (C=8)",      lambda: SNNPolicy(C=8),   "snn_policy_best.pt"),
    ("SNN C=2",             lambda: SNNPolicy(C=2),   "snn_C2_best.pt"),
    ("SNN C=4",             lambda: SNNPolicy(C=4),   "snn_C4_best.pt"),
    ("SNN C=8",             lambda: SNNPolicy(C=8),   "snn_C8_best.pt"),
    ("SNN C=16",            lambda: SNNPolicy(C=16),  "snn_C16_best.pt"),
]

rows = []
for label, make, ckpt in models:
    try:
        m = make()
        m.load_state_dict(torch.load(ckpt))
        m.eval()
    except FileNotFoundError:
        print(f"[skip] {label}: {ckpt} not found")
        continue
    print(f"{label}:")
    rate = evaluate(m, num_episodes=N, greedy=False, temperature=T)   # prints its own line
    rows.append((label, ckpt, rate))

# save a clean table for the report
with open("data/eval_final.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "checkpoint", f"success_rate_T{T}_N{N}"])
    for label, ckpt, rate in rows:
        w.writerow([label, ckpt, f"{rate:.4f}"])

print("\n=== FINAL (500 goals, T=0.3) ===")
for label, _, rate in rows:
    print(f"  {label:22s} {rate:.1%}")
print("saved eval_final.csv")

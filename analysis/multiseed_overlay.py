"""overlay the mlp and snn multi-seed curves."""
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotstyle; plotstyle.apply()


def load(path):
    ep, mean, std = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["episode"]))
            mean.append(float(row["mean_success"]) * 100)
            std.append(float(row["std_success"]) * 100)
    return np.array(ep), np.array(mean), np.array(std)


mlp = load("data/multiseed_mlp_5seeds.csv")
snn = load("data/multiseed_snn_3seeds.csv")

plt.figure(figsize=(7, 4.5))
for (ep, m, s), color, label in [(mlp, "tab:blue", "MLP (5 seeds)"),
                                 (snn, "tab:orange", "SNN (3 seeds)")]:
    plt.plot(ep, m, color=color, lw=2, label=label)
    plt.fill_between(ep, m - s, m + s, color=color, alpha=0.2)
plt.xlabel("episode"); plt.ylabel("success rate (%)  @ T=0.3")
plt.title("MLP vs SNN: success over training (mean $\\pm$ 1 std)")
plt.legend()
plt.savefig("figures/multiseed_overlay.pdf", dpi=120); plt.close()
print("saved figures/multiseed_overlay.pdf")

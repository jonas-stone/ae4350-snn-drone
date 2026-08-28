"""firing-rate histogram + summary from csv."""
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---- edit ----
FIGSIZE     = (7, 4.5)
TITLE_SIZE  = 15
LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 12
BINS        = 20
OUT         = "figures/firing_stats.pdf"
# --------------
import plotstyle; plotstyle.apply()

rates, C = [], 8
with open("data/firing_rates.csv") as f:
    for r in csv.DictReader(f):
        rates.append(float(r["avg_rate"])); C = int(r["C"])
rates = np.array(rates)

# ---- summary (derivable from the per-neuron rates) ----
mean_rate = rates.mean()
spikes_per_decision = rates.sum() * C           # sum of rates * cycles
dead = np.mean(rates < 0.01)
saturated = np.mean(rates > 0.99)
print(f"neurons:              {len(rates)}")
print(f"mean firing rate:     {mean_rate:.1%}")
print(f"spikes per decision:  {spikes_per_decision:.1f}")
print(f"ANN dense activations:{len(rates)} (every neuron, every pass)")
print(f"dead neurons (<1%):   {dead:.1%}")
print(f"saturated (>99%):     {saturated:.1%}")

plt.figure(figsize=FIGSIZE)
plt.hist(rates, bins=BINS, color="tab:purple", edgecolor="k")
plt.axvline(mean_rate, color="red", ls="--", label=f"mean {mean_rate:.1%}")
plt.xlabel("per-neuron average firing rate", fontsize=LABEL_SIZE)
plt.ylabel("number of hidden neurons", fontsize=LABEL_SIZE)
plt.title(f"firing-rate distribution across the {len(rates)} hidden neurons", fontsize=TITLE_SIZE)
plt.xticks(fontsize=TICK_SIZE); plt.yticks(fontsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.tight_layout(); plt.savefig(OUT, dpi=120); plt.close()
print(f"saved {OUT}")

"""draw success vs cycle count from csv."""
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---- edit ----
FIGSIZE     = (7, 4.5)
TITLE_SIZE  = 15
LABEL_SIZE  = 14
TICK_SIZE   = 12
OUT         = "figures/csweep.pdf"
# --------------
import plotstyle; plotstyle.apply()

Cs, rates = [], []
with open("data/csweep_final.csv") as f:
    for r in csv.DictReader(f):
        Cs.append(int(r["C"])); rates.append(float(r["success_500"]) * 100)
order = np.argsort(Cs)
Cs = [Cs[i] for i in order]; rates = [rates[i] for i in order]
x = np.arange(len(Cs))                       # even spacing for 2, 4, 8, 16

plt.figure(figsize=FIGSIZE)
plt.plot(x, rates, marker="o", ms=9, color="tab:blue")
plt.xticks(x, Cs, fontsize=TICK_SIZE); plt.yticks(fontsize=TICK_SIZE)
plt.ylim(0, 100); #plt.xlim(-0.5, len(Cs) - 0.5)
plt.xlabel("integration cycles $C$", fontsize=LABEL_SIZE)
plt.ylabel("success rate (%), 500 goals @ T=0.3", fontsize=LABEL_SIZE)
plt.title("SNN performance vs integration cycles", fontsize=TITLE_SIZE)
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=120); plt.close()
print(f"saved {OUT}")

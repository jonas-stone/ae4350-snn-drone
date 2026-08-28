"""draw firing rate vs threshold from csv."""
import csv
import matplotlib.pyplot as plt
import plotstyle; plotstyle.apply()

# ---- edit ----
FIGSIZE = (7, 4.5)
OUT     = "figures/threshold_sweep.pdf"
# --------------

th, rate = [], []
with open("data/threshold_sweep.csv") as f:
    for r in csv.DictReader(f):
        th.append(float(r["threshold"])); rate.append(float(r["mean_firing_rate"]) * 100)

plt.figure(figsize=FIGSIZE)
plt.axhspan(10, 30, color="tab:green", alpha=0.12, label="healthy 10--30\\%")
plt.plot(th, rate, marker="o", color="tab:blue")
plt.axvline(0.3, color="tab:red", ls="--", label="chosen $V_{th}=0.3$")
plt.xlabel("LIF threshold $V_{th}$")
plt.ylabel("mean firing rate at init (%)")
plt.title("firing rate vs threshold (at initialisation)")
plt.legend()
plt.savefig(OUT, dpi=120); plt.close()
print(f"saved {OUT}")

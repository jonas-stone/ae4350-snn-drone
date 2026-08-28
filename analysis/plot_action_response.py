"""draw scene | encoding | behaviour from csv."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---- edit ----
FIGSIZE     = (11, 3.8)
TITLE_SIZE  = 15
PANEL_SIZE  = 15
LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 12
OUT         = "figures/action_response.pdf"
# --------------
import plotstyle; plotstyle.apply()

# ---- load curves ----
dist, l1, l2 = [], [], []
P = {"p_up": [], "p_down": [], "p_left": [], "p_right": []}
with open("data/action_response.csv") as f:
    for r in csv.DictReader(f):
        dist.append(float(r["distance"])); l1.append(float(r["l1_rate"])); l2.append(float(r["l2_rate"]))
        for k in P: P[k].append(float(r[k]))
dist = np.array(dist)

# ---- load scene ----
wall = start = end = None; rays = []
with open("data/action_scene.csv") as f:
    for r in csv.DictReader(f):
        vals = [float(r["a"]), float(r["b"]), float(r["c"]), float(r["d"])]
        if r["type"] == "wall":  wall = vals
        elif r["type"] == "start": start = vals[:2]
        elif r["type"] == "end":   end = vals[:2]
        elif r["type"] == "ray":   rays.append(vals)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=FIGSIZE)

# scene
ax0.add_patch(Rectangle((wall[0], wall[1]), wall[2], wall[3], color="gray"))
for x1, y1, x2, y2 in rays:
    ax0.plot([x1, x2], [y1, y2], color="tab:red", lw=0.9, alpha=0.6)
ax0.plot(*start, "o", color="tab:green", ms=11, label="start (far)")
ax0.plot(*end, "o", color="tab:red", ms=11, label="end (near wall)")
ax0.annotate("", xy=(end[0], end[1] - 0.4), xytext=(start[0], start[1] + 0.4),
             arrowprops=dict(arrowstyle="->", lw=1.5))
ax0.set_xlim(0, 10); ax0.set_ylim(0, 10); ax0.set_aspect("equal")
ax0.set_title("scene: drone marches into a wall", fontsize=PANEL_SIZE)
ax0.tick_params(labelsize=TICK_SIZE); ax0.legend(fontsize=LEGEND_SIZE, loc="lower left")

# encoding
ax1.plot(dist, l1, marker="o", ms=3, label="hidden layer 1")
ax1.plot(dist, l2, marker="s", ms=3, label="hidden layer 2")
ax1.invert_xaxis()
ax1.set_xlabel("distance to wall ahead  (approaching ->)", fontsize=LABEL_SIZE)
ax1.set_ylabel("mean firing rate", fontsize=LABEL_SIZE)
ax1.set_title("encoding: spike activity", fontsize=PANEL_SIZE)
ax1.tick_params(labelsize=TICK_SIZE); ax1.legend(fontsize=LEGEND_SIZE); ax1.grid(alpha=0.3)

# behaviour
labels = {"p_up": "up (into wall)", "p_down": "down (retreat)", "p_left": "left", "p_right": "right"}
for k, lab in labels.items():
    ax2.plot(dist, P[k], marker="o", ms=3, label=lab)
ax2.invert_xaxis()
ax2.set_xlabel("distance to wall ahead  (approaching ->)", fontsize=LABEL_SIZE)
ax2.set_ylabel("action probability", fontsize=LABEL_SIZE)
ax2.set_title("behaviour: decoded action", fontsize=PANEL_SIZE)
ax2.tick_params(labelsize=TICK_SIZE); ax2.legend(fontsize=LEGEND_SIZE); ax2.grid(alpha=0.3)

fig.suptitle("the SNN encodes obstacle proximity and acts to avoid it", fontsize=TITLE_SIZE)
plt.tight_layout(); plt.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close()
print(f"saved {OUT}")

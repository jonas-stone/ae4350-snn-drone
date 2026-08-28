"""draw the mlp vs snn trajectory grid from csv."""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---- edit ----
FIGSIZE      = (9, 6)      # overall figure size (inches)
NROWS, NCOLS = 2, 3         # panel grid (must hold all seeds)
TITLE_SIZE   = 15           # suptitle
PANEL_SIZE   = 12           # per-seed titles
LABEL_SIZE   = 14
TICK_SIZE    = 8
LEGEND_SIZE  = 8
LINEWIDTH    = 1.5
GOAL_RADIUS  = 0.5
COLORS       = {"MLP": "tab:blue", "SNN": "tab:orange"}
OUT          = "figures/trajectory_compare.pdf"
# --------------
import plotstyle; plotstyle.apply()
# ---- load obstacles ----
obstacles = []
with open("data/obstacles.csv") as f:
    for r in csv.DictReader(f):
        obstacles.append((float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])))

# ---- load trajectories ----
paths   = defaultdict(lambda: {"x": [], "y": []})   # (seed, net) -> path
reached = {}                                         # (seed, net) -> bool
goal    = {}                                         # seed -> (x, y)
start   = {}                                         # seed -> (x, y)
with open("data/trajectories.csv") as f:
    for r in csv.DictReader(f):
        seed, net = int(r["seed"]), r["net"]
        paths[(seed, net)]["x"].append(float(r["x"]))
        paths[(seed, net)]["y"].append(float(r["y"]))
        reached[(seed, net)] = bool(int(r["reached"]))
        goal[seed]  = (float(r["goal_x"]), float(r["goal_y"]))
        start[seed] = (float(r["start_x"]), float(r["start_y"]))

seeds = sorted(goal)

fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
for ax, seed in zip(axes.flat, seeds):
    for x, y, w, h in obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color="lightgray"))
    ax.add_patch(Circle(goal[seed], GOAL_RADIUS, color="green", alpha=0.5))
    ax.plot(*start[seed], "ks", ms=6)
    for net in ["MLP", "SNN"]:
        d, ok = paths[(seed, net)], reached[(seed, net)]
        ax.plot(d["x"], d["y"], "-", color=COLORS[net], lw=LINEWIDTH,
                label=f"{net} {'reached' if ok else 'failed'}")
        ax.plot(d["x"][-1], d["y"][-1], "o" if ok else "x", color=COLORS[net], ms=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(f"seed {seed}", fontsize=PANEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right")

# hide any unused panels
for ax in axes.flat[len(seeds):]:
    ax.axis("off")

fig.suptitle("MLP vs SNN trajectories on identical goals (T=0.3)", fontsize=TITLE_SIZE)
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
plt.close()
print(f"saved {OUT}")

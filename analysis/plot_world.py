"""render the environment (obstacles, centre start, goal, rays)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import numpy as np
import matplotlib.pyplot as plt
import plotstyle; plotstyle.apply()
from matplotlib.patches import Rectangle, Circle
from drone_env import DroneNavEnv

env = DroneNavEnv()
env.reset(seed=2)                       # fixes an example goal in free space
env.drone_pos = env.start.copy()        # draw sensors from the (centre) start

fig, ax = plt.subplots(figsize=(5, 5))   # embed at 0.5\textwidth -> text matches the 7-wide @0.7 figures
ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal"); ax.grid(False)
for s in ax.spines.values():          # draw all four borders: the arena is a closed box
    s.set_visible(True)

# obstacles
for i, (x, y, w, h) in enumerate(env.obstacles):
    ax.add_patch(Rectangle((x, y), w, h, color="gray",
                           label="obstacle" if i == 0 else None))

# 8 ray sensors from the start
dirs = env._ray_directions(); dists = env._get_ray_distances()
for j, (dvec, dd) in enumerate(zip(dirs, dists)):
    end = env.drone_pos + dd * dvec
    ax.plot([env.start[0], end[0]], [env.start[1], end[1]], color="tab:red",
            lw=0.9, alpha=0.7, label="ray sensor" if j == 0 else None)

# start, goal
ax.plot(env.start[0], env.start[1], "s", color="tab:blue", ms=11, label="start (centre)")
ax.add_patch(Circle(env.goal, env.goal_radius, color="tab:green", label="goal"))

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("navigation environment")
ax.legend(loc="upper left")
plt.savefig("figures/world.pdf", dpi=150); plt.close()
print("saved figures/world.pdf")

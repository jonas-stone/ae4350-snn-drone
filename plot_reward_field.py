import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from drone_env import DroneNavEnv

SEED = 138                          # same goal you watched flicker
env = DroneNavEnv()
env.reset(seed=SEED)                # fixes the goal + builds env._path_field (BFS)

W, H, res = env.width, env.height, env.grid_res

# --- field 1: the BFS path-distance field the reward ACTUALLY uses ---
bfs = env._path_field.copy()                 # (nx, ny), inf where blocked
bfs[np.isinf(bfs)] = np.nan                  # blocked cells -> blank on the heatmap

# --- field 2: the SAME BFS field, but through a concave sqrt funnel ---
# sqrt makes the slope steeper near the goal -> stronger pull for the final approach
bfs_funnel = np.sqrt(bfs)                     # nan stays nan (blocked cells)

def draw(ax, field, title):
    # field is (nx, ny) = (x, y); imshow wants (row, col) = (y, x) -> transpose
    im = ax.imshow(field.T, origin="lower", extent=[0, W, 0, H],
                   cmap="viridis", interpolation="nearest")
    for x, y, w, h in env.obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color="lightgray", ec="k", lw=0.5))
    ax.add_patch(Circle(env.goal, env.goal_radius, color="red", fill=False, lw=2))
    ax.plot(env.start[0], env.start[1], "ws", mec="k", markersize=8)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label="distance to goal")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
draw(a1, bfs,        "BFS path-distance, LINEAR (current reward)")
draw(a2, bfs_funnel, "BFS path-distance, SQRT funnel (proposed)")
fig.suptitle(f"reward field, seed {SEED}   (goal = red circle, start = white square)")
plt.tight_layout()
plt.savefig("reward_field.png", dpi=120)
plt.close(fig)
print("saved reward_field.png")

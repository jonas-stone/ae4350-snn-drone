"""MLP vs SNN on the SAME goals: draw both policies' paths on one map per seed,
so you can SEE whether they navigate alike or differ. Deployment protocol T=0.3."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from drone_env import DroneNavEnv
from networks import MLPPolicy, SNNPolicy

T = 0.3
SEEDS = [0, 1, 2, 3, 4, 5]          # the goals to compare on

mlp = MLPPolicy(); mlp.load_state_dict(torch.load("mlp_policy_best.pt")); mlp.eval()
snn = SNNPolicy(C=8); snn.load_state_dict(torch.load("snn_policy_best.pt")); snn.eval()


def rollout(policy, seed):
    env = DroneNavEnv()
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)                       # make the sampled path reproducible
    traj, done, info = [env.drone_pos.copy()], False, {}
    while not done:
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs), greedy=False, temperature=T)
        obs, _, term, trunc, info = env.step(a.item())
        traj.append(env.drone_pos.copy())
        done = term or trunc
    return np.array(traj), bool(info.get("reached_goal", False)), env


fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, seed in zip(axes.flat, SEEDS):
    mlp_traj, mlp_ok, env = rollout(mlp, seed)
    snn_traj, snn_ok, _   = rollout(snn, seed)

    for x, y, w, h in env.obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color="lightgray"))
    ax.add_patch(Circle(env.goal, env.goal_radius, color="green", alpha=0.5))
    ax.plot(env.start[0], env.start[1], "ks", markersize=8)

    ax.plot(mlp_traj[:, 0], mlp_traj[:, 1], "-", color="tab:blue", lw=1.6,
            label=f"MLP {'✓' if mlp_ok else '✗'}")
    ax.plot(snn_traj[:, 0], snn_traj[:, 1], "-", color="tab:orange", lw=1.6,
            label=f"SNN {'✓' if snn_ok else '✗'}")
    ax.plot(mlp_traj[-1, 0], mlp_traj[-1, 1], "o" if mlp_ok else "x", color="tab:blue", ms=9)
    ax.plot(snn_traj[-1, 0], snn_traj[-1, 1], "o" if snn_ok else "x", color="tab:orange", ms=9)

    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
    ax.set_title(f"seed {seed}")
    ax.legend(loc="upper left", fontsize=8)

fig.suptitle("MLP vs SNN trajectories on identical goals (T=0.3)")
plt.tight_layout()
plt.savefig("trajectory_compare.png", dpi=120)
plt.close(fig)
print("saved trajectory_compare.png")

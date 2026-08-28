"""roll out the mlp and snn and dump trajectories + obstacles to csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import torch
import numpy as np
from drone_env import DroneNavEnv
from networks import MLPPolicy, SNNPolicy

SEEDS = [0, 1, 2, 3, 4, 5]
T = 0.3

mlp = MLPPolicy(); mlp.load_state_dict(torch.load("mlp_policy_best.pt")); mlp.eval()
snn = SNNPolicy(C=8); snn.load_state_dict(torch.load("snn_policy_best.pt")); snn.eval()


def rollout(policy, seed):
    env = DroneNavEnv()
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)
    traj, done, info = [env.drone_pos.copy()], False, {}
    while not done:
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs), greedy=False, temperature=T)
        obs, _, term, trunc, info = env.step(a.item())
        traj.append(env.drone_pos.copy())
        done = term or trunc
    return np.array(traj), bool(info.get("reached_goal", False)), env.goal.copy(), env.start.copy()

os.makedirs("data", exist_ok=True)

# fixed obstacle layout
with open("data/obstacles.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x", "y", "w", "h"])
    for row in DroneNavEnv().obstacles:
        w.writerow([float(v) for v in row])

# trajectories
with open("data/trajectories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "net", "reached", "start_x", "start_y", "goal_x", "goal_y", "step", "x", "y"])
    for seed in SEEDS:
        for name, pol in [("MLP", mlp), ("SNN", snn)]:
            traj, reached, goal, start = rollout(pol, seed)
            for i, (x, y) in enumerate(traj):
                w.writerow([seed, name, int(reached), start[0], start[1],
                            goal[0], goal[1], i, x, y])
print("saved data/trajectories.csv and data/obstacles.csv")

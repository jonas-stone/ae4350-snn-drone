"""march the drone into a wall; dump firing rates and action probs to csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import core modules when run from the project root
import os
import csv
import torch
import numpy as np
from networks import SNNPolicy
from drone_env import DroneNavEnv

p = SNNPolicy(C=8)
p.load_state_dict(torch.load("snn_policy_best.pt")); p.eval()


@torch.no_grad()
def probe(obs):
    x = torch.as_tensor(obs, dtype=torch.float32)
    m1 = p.lif1.init_leaky(); m2 = p.lif2.init_leaky(); s1 = s2 = 0.0
    for _ in range(p.C):
        spk1, m1 = p.lif1(p.fc1(x), m1); s1 += spk1.sum().item()
        spk2, m2 = p.lif2(p.fc2(spk1), m2); s2 += spk2.sum().item()
    pr = torch.softmax(p.forward(x), dim=-1).numpy()
    return s1 / (p.C * 64), s2 / (p.C * 64), pr


env = DroneNavEnv()
env.obstacles = np.array([[1.0, 9.0, 8.0, 0.9]], dtype=np.float32)
env.heading = 0.0
ys = np.linspace(1.0, 8.7, 30)
os.makedirs("data", exist_ok=True)

with open("data/action_response.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["distance", "l1_rate", "l2_rate", "p_up", "p_down", "p_left", "p_right"])
    for y in ys:
        env.drone_pos = np.array([5.0, y], dtype=np.float32)
        env.goal = env.drone_pos + np.array([0.0, 1.0], dtype=np.float32)
        a, b, pr = probe(env._get_observation())
        w.writerow([9.0 - y, a, b, pr[0], pr[1], pr[2], pr[3]])

# scene geometry: wall (x,y,w,h), start, end, and the rays at the end position
env.drone_pos = np.array([5.0, ys[-1]], dtype=np.float32)
with open("data/action_scene.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type", "a", "b", "c", "d"])
    w.writerow(["wall", 1.0, 9.0, 8.0, 0.9])
    w.writerow(["start", 5.0, ys[0], 0, 0])
    w.writerow(["end", 5.0, ys[-1], 0, 0])
    for dvec, dd in zip(env._ray_directions(), env._get_ray_distances()):
        e = env.drone_pos + dd * dvec
        w.writerow(["ray", 5.0, ys[-1], e[0], e[1]])
print("saved data/action_response.csv and data/action_scene.csv")

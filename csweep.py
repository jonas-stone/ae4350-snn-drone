"""C-cycle sweep: how much 'thinking time' does the SNN need?
Trains a SEPARATE SNN for each number of integration cycles C, then plots best
success vs C.  EXPENSIVE: each C is a full training run.  Run overnight."""
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from drone_env import DroneNavEnv
from networks import SNNPolicy, ValueNet
from train import run_episode, compute_returns, evaluate

C_VALUES = [2, 4, 8, 16]        # cycles to test (each = one full training run)
EPISODES = 3000                 # fewer than the main run, to keep the sweep feasible
EVAL_EVERY = 300


def train_snn(C, num_episodes=EPISODES, gamma=0.99, lr=1e-3):
    """Actor-critic training for an SNN with a given cycle count C. Returns best T=0.3 success."""
    env = DroneNavEnv()
    policy = SNNPolicy(C=C)
    critic = ValueNet()
    actor_opt  = torch.optim.Adam(policy.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
    best_rate = -1.0
    t0 = time.time()

    for episode in range(num_episodes):
        log_probs, rewards, states = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)
        values = critic(torch.stack(states))
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss  = -(torch.stack(log_probs) * advantages).sum()
        critic_loss = ((values - returns) ** 2).mean()
        actor_opt.zero_grad();  actor_loss.backward();  actor_opt.step()
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        if (episode + 1) % EVAL_EVERY == 0:
            rate = evaluate(policy, num_episodes=100, greedy=False, temperature=0.3)
            if rate > best_rate:
                best_rate = rate
                torch.save(policy.state_dict(), f"snn_C{C}_best.pt")
            print(f"    C={C} | ep {episode+1} | success {rate:.0%} | best {best_rate:.0%} | {time.time()-t0:.0f}s")

    return best_rate


results = {}
for C in C_VALUES:
    print(f"=== training SNN with C={C} ===")
    results[C] = train_snn(C)
    print(f"=== C={C} best: {results[C]:.0%} ===\n")

# --- plot success vs C ---
Cs = list(results.keys())
rates = [results[c] for c in Cs]
plt.figure(figsize=(7, 5))
plt.plot(Cs, [r * 100 for r in rates], marker="o", ms=8)
plt.xlabel("integration cycles C (thinking time per decision)")
plt.ylabel("best success rate (%)  @ T=0.3")
plt.title("SNN performance vs number of integration cycles")
plt.xticks(Cs)
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("csweep.png", dpi=120); plt.close()
print("saved csweep.png")
print("results:", {c: f"{r:.0%}" for c, r in results.items()})

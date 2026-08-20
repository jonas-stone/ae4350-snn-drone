import torch
from drone_env import DroneNavEnv
from networks import MLPPolicy
import time
import numpy as np


def train(num_episodes=200, gamma=0.99, lr=1e-3):
    env = DroneNavEnv()
    policy = MLPPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    start = time.time()

    for episode in range(num_episodes):
        log_probs, rewards = run_episode(env, policy)     # NO seed -> new obstacles each time
        returns = compute_returns(rewards, gamma)
        loss = compute_loss(log_probs, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))

        if (episode + 1) % 100 == 0:
            avg = sum(reward_history[-100:]) / 100
            elapsed = time.time() - start
            print(f"episode {episode+1:4d} | avg reward (last 100): {avg:7.3f} | {elapsed:5.1f}s")

    return policy, reward_history




def run_episode(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    log_probs = []
    rewards = []
    done = False

    while not done:
        obs_t = torch.from_numpy(obs)              # numpy obs -> tensor
        action, log_prob = policy.act(obs_t)       # stochastic action + its log-prob
        obs, reward, terminated, truncated, _ = env.step(action.item())

        log_probs.append(log_prob)                 # keep as a TENSOR (carries grad_fn)
        rewards.append(reward)                     # keep as a plain float
        done = terminated or truncated

    return log_probs, rewards

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G          # this reward + discounted future
        returns.insert(0, G)       # prepend, so order matches the timesteps
    returns = torch.tensor(returns, dtype=torch.float32)

    # standardize for stable, well-scaled gradients
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def compute_loss(log_probs, returns):
    log_probs = torch.stack(log_probs)      # list of scalar tensors -> one 1D tensor
    loss = -(log_probs * returns).sum()     # -Σ logπ(aₜ)·Gₜ
    return loss

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def plot_rewards(reward_history, window=20, save_path="learning_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(reward_history, alpha=0.3, label="per episode")   # faint raw curve

    # smoothed moving average, so the trend is readable through the noise
    smoothed = [sum(reward_history[i-window:i]) / window
                for i in range(window, len(reward_history) + 1)]
    plt.plot(range(window, len(reward_history) + 1), smoothed, label=f"{window}-episode average")

    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def evaluate(policy, num_episodes=100, render_path=None):
    env = DroneNavEnv()
    successes = 0
    last_traj, last_obs = None, None

    for ep in range(num_episodes):
        obs, _ = env.reset()
        traj = [env.drone_pos.copy()]
        done = False
        info = {}
        while not done:
            with torch.no_grad():                              # no gradients needed at eval
                action, _ = policy.act(torch.from_numpy(obs), greedy=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            traj.append(env.drone_pos.copy())
            done = terminated or truncated
        if info.get("reached_goal"):
            successes += 1
        last_traj, last_obs = np.array(traj), env.obstacles.copy()

    rate = successes / num_episodes
    print(f"success rate: {successes}/{num_episodes} = {rate:.0%}")

    if render_path is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')

        for x, y, w, h in last_obs:                     # obstacles of the last episode
            ax.add_patch(Rectangle((x, y), w, h, color='gray'))

        ax.add_patch(Circle(env.goal, env.goal_radius, color='green'))
        ax.plot(env.start[0], env.start[1], 'gs', label='start')
        ax.plot(last_traj[:, 0], last_traj[:, 1], '-', color='tab:blue',
                linewidth=1.5, label='path')
        ax.plot(last_traj[-1, 0], last_traj[-1, 1], 'x', color='red',
                markersize=10, label='end')
        ax.legend(loc='upper left')

        plt.savefig(render_path)
        plt.close(fig)

    return rate







if __name__ == "__main__":
    policy, history = train(num_episodes=2000)
    torch.save(policy.state_dict(), "mlp_policy.pt")
    plot_rewards(history)
    evaluate(policy, num_episodes=200, render_path="trajectory.png")
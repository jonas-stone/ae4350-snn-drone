import torch
from drone_env import DroneNavEnv
from networks import MLPPolicy, SNNPolicy, ValueNet
import time
import numpy as np


def train(train_which='SNN', num_episodes=3000, gamma=0.99, lr=1e-3, eval_every=300, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = DroneNavEnv()
    if seed is not None:
        env.reset(seed=seed)        # seed the goal-sampling stream; stays deterministic after
    policy = MLPPolicy() if train_which == 'MLP' else SNNPolicy()

    critic = ValueNet()
    actor_opt  = torch.optim.Adam(policy.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    reward_history = []
    success_history = []
    best_rate = -1.0
    start = time.time()

    for episode in range(num_episodes):
        log_probs, rewards, states = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)

        values = critic(torch.stack(states))
        advantages = returns - values.detach()          # advantage = return minus baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss  = -(torch.stack(log_probs) * advantages).sum()
        critic_loss = ((values - returns) ** 2).mean()
        actor_opt.zero_grad();  actor_loss.backward();  actor_opt.step()
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        reward_history.append(sum(rewards))

        if (episode + 1) % eval_every == 0:
            rate = evaluate(policy, num_episodes=100, greedy=False, temperature=0.3)
            if rate > best_rate:
                best_rate = rate
                torch.save(policy.state_dict(),
                           "mlp_policy_best.pt" if train_which == 'MLP' else "snn_policy_best.pt")
            success_history.append((episode + 1, rate))
            print(f"  -> episode {episode+1} | success {rate:.0%} | best {best_rate:.0%} | {time.time()-start:.0f}s")

    print(f"best success rate: {best_rate:.0%}")
    return policy, reward_history, success_history


def run_episode(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    states, log_probs, rewards = [], [], []
    done = False
    while not done:
        obs_t = torch.from_numpy(obs)
        action, log_prob = policy.act(obs_t)            # sampled action + its log-prob
        obs, reward, terminated, truncated, _ = env.step(action.item())
        log_probs.append(log_prob)                      # tensor (carries grad)
        rewards.append(reward)
        states.append(obs_t)
        done = terminated or truncated
    return log_probs, rewards, states


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def plot_rewards(reward_history, window=20, save_path="learning_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(reward_history, alpha=0.3, label="per episode")
    smoothed = [sum(reward_history[i-window:i]) / window
                for i in range(window, len(reward_history) + 1)]
    plt.plot(range(window, len(reward_history) + 1), smoothed, label=f"{window}-episode average")
    plt.xlabel("episode"); plt.ylabel("total reward"); plt.legend()
    plt.savefig(save_path); plt.close()


def evaluate(policy, num_episodes=100, render_path=None, greedy=True, temperature=1.0, watch=False, watch_delay=0.002):
    env = DroneNavEnv()
    if watch:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

    successes = 0
    last_traj, last_obs = None, None
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)                     # seeded: every policy faces the same goals

        if watch:
            ax.clear()
            ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')
            for x, y, w, h in env.obstacles:
                ax.add_patch(Rectangle((x, y), w, h, color='gray'))
            ax.add_patch(Circle(env.goal, env.goal_radius, color='green'))
            ax.plot(env.start[0], env.start[1], 'ks', markersize=8)
            dot,  = ax.plot([], [], 'bo', markersize=8)
            trail, = ax.plot([], [], '-', color='tab:blue', lw=1)
            xs, ys = [], []

        traj = [env.drone_pos.copy()]
        done = False
        info = {}
        while not done:
            with torch.no_grad():
                action, _ = policy.act(torch.from_numpy(obs), greedy=greedy, temperature=temperature)
            obs, reward, terminated, truncated, info = env.step(action.item())
            traj.append(env.drone_pos.copy())

            if watch:
                xs.append(env.drone_pos[0]); ys.append(env.drone_pos[1])
                dot.set_data([env.drone_pos[0]], [env.drone_pos[1]])
                trail.set_data(xs, ys)
                ax.set_title(f"seed {ep}")
                plt.pause(watch_delay)

            done = terminated or truncated

        if watch:
            ax.set_title(f"seed {ep} — {'REACHED' if info.get('reached_goal') else 'FAILED'}")
            plt.pause(0.6)

        if info.get("reached_goal"):
            successes += 1
        last_traj, last_obs = np.array(traj), env.obstacles.copy()

    rate = successes / num_episodes
    print(f"success rate: {successes}/{num_episodes} = {rate:.0%}")

    if render_path is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')
        for x, y, w, h in last_obs:
            ax.add_patch(Rectangle((x, y), w, h, color='gray'))
        ax.add_patch(Circle(env.goal, env.goal_radius, color='green'))
        ax.plot(env.start[0], env.start[1], 'gs', label='start')
        ax.plot(last_traj[:, 0], last_traj[:, 1], '-', color='tab:blue', linewidth=1.5, label='path')
        ax.plot(last_traj[-1, 0], last_traj[-1, 1], 'x', color='red', markersize=10, label='end')
        ax.legend(loc='upper left')
        plt.savefig(render_path); plt.close(fig)

    return rate


def plot_multi_goal(policy, seeds, save_path="multi_goal.png", greedy=True):
    env = DroneNavEnv()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect('equal')
    for x, y, w, h in env.obstacles:
        ax.add_patch(Rectangle((x, y), w, h, color='gray'))
    ax.plot(env.start[0], env.start[1], 'ks', markersize=9, label='start')

    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    reached_count = 0
    for seed, c in zip(seeds, colors):
        obs, _ = env.reset(seed=seed)
        traj = [env.drone_pos.copy()]
        done, info = False, {}
        while not done:
            with torch.no_grad():
                action, _ = policy.act(torch.from_numpy(obs), greedy=greedy)
            obs, _, terminated, truncated, info = env.step(action.item())
            traj.append(env.drone_pos.copy())
            done = terminated or truncated
        traj = np.array(traj)
        reached = bool(info.get("reached_goal", False))
        reached_count += reached
        ax.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=1.5)
        ax.add_patch(Circle(env.goal, env.goal_radius, color=c, alpha=0.4))
        ax.plot(traj[-1, 0], traj[-1, 1], 'o' if reached else 'x', color=c, markersize=10)

    ax.set_title(f"{reached_count}/{len(seeds)} goals reached")
    ax.legend(loc='upper left')
    plt.savefig(save_path); plt.close(fig)


def save_history(tag, reward_history, success_history):
    import csv, os
    os.makedirs("data", exist_ok=True)
    with open(f"data/{tag}_reward_history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "episode_reward"])
        for i, r in enumerate(reward_history):
            w.writerow([i + 1, r])
    with open(f"data/{tag}_success_history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "success_T0.3"])
        for ep, rate in success_history:
            w.writerow([ep, rate])
    print(f"saved data/{tag}_reward_history.csv and data/{tag}_success_history.csv")


if __name__ == "__main__":
    train_which = 'SNN'
    tag = train_which.lower()
    policy, reward_history, success_history = train(train_which=train_which, num_episodes=6000)
    save_history(tag, reward_history, success_history)
    import os; os.makedirs("figures", exist_ok=True)
    plot_rewards(reward_history, save_path=f"figures/{tag}_learning_curve.png")
    best = SNNPolicy() if train_which == 'SNN' else MLPPolicy()
    best.load_state_dict(torch.load(f"{tag}_policy_best.pt"))
    evaluate(best, num_episodes=200, render_path=f"figures/{tag}_trajectory.png")
    plot_multi_goal(best, seeds=range(10), save_path=f"figures/{tag}_multi_goal.png")

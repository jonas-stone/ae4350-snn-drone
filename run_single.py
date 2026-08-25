import torch
from networks import MLPPolicy, SNNPolicy
from drone_env import DroneNavEnv

# load both trained models
mlp = MLPPolicy()
mlp.load_state_dict(torch.load("mlp_policy_best.pt"))

snn = SNNPolicy()
snn.load_state_dict(torch.load("snn_policy_best.pt"))

env = DroneNavEnv()
policy = mlp                 # diagnosing the MLP
BAD_SEED = 138

# run the SAME bad seed greedy vs stochastic, to test the limit-cycle hypothesis
for greedy in (True, False):
    obs, _ = env.reset(seed=BAD_SEED)
    done, info, steps = False, {}, 0
    while not done:
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs), greedy=greedy)
        obs, _, term, trunc, info = env.step(a.item())
        done = term or trunc; steps += 1
    print(f"greedy={greedy}: reached={info.get('reached_goal')} in {steps} steps")
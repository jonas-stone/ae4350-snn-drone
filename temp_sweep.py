import torch
from networks import MLPPolicy
from train import evaluate

mlp = MLPPolicy()
mlp.load_state_dict(torch.load("mlp_policy_best.pt"))

N = 200
print(f"greedy (argmax):")
evaluate(mlp, num_episodes=N, greedy=True)

for T in (1.0, 0.5, 0.3, 0.2, 0.1):
    print(f"stochastic, T={T}:")
    evaluate(mlp, num_episodes=N, greedy=False, temperature=T)

import torch
from networks import MLPPolicy, SNNPolicy
from train import evaluate

# load both trained models
mlp = MLPPolicy()
mlp.load_state_dict(torch.load("mlp_policy_best.pt"))

snn = SNNPolicy()
snn.load_state_dict(torch.load("snn_policy_best.pt"))

N = 200  # how many identical goals to test both on

print("=== MLP ===")
mlp_rate = evaluate(policy=mlp, num_episodes=N, watch=True, greedy=False)      # <- fill in

print("=== SNN ===")
snn_rate = evaluate(policy=snn, num_episodes=N, watch=True, greedy=False)      # <- fill in

print(f"\nHEAD-TO-HEAD on the SAME {N} goals:")
print(f"  MLP: {mlp_rate:.1%}")
print(f"  SNN: {snn_rate:.1%}")
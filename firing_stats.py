"""How sparse is the SNN?  The efficiency argument: an ANN activates every
neuron on every pass (dense); an SNN neuron only emits a spike some fraction of
the time (sparse), and neuromorphic energy scales with the number of spikes."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from networks import SNNPolicy
from drone_env import DroneNavEnv

p = SNNPolicy()
p.load_state_dict(torch.load("snn_policy_best.pt"))
p.eval()


@torch.no_grad()
def layer_rates(obs):
    """Per-neuron firing rate (spikes / C) for each hidden layer, one state."""
    x = torch.as_tensor(obs, dtype=torch.float32)
    mem1 = p.lif1.init_leaky()
    mem2 = p.lif2.init_leaky()
    s1 = torch.zeros(p.fc1.out_features)
    s2 = torch.zeros(p.fc2.out_features)
    for _ in range(p.C):
        spk1, mem1 = p.lif1(p.fc1(x), mem1); s1 += spk1
        spk2, mem2 = p.lif2(p.fc2(spk1), mem2); s2 += spk2
    return (s1 / p.C).numpy(), (s2 / p.C).numpy()


# --- collect real states ---
env = DroneNavEnv()
states = []
for seed in range(40):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        states.append(obs.copy())
        a, _ = p.act(torch.from_numpy(obs), greedy=False, temperature=0.3)
        obs, _, term, trunc, _ = env.step(a.item())
        done = term or trunc
states = np.array(states)

R1 = np.array([layer_rates(s)[0] for s in states])   # (N, 64) firing rates, layer 1
R2 = np.array([layer_rates(s)[1] for s in states])   # (N, 64) firing rates, layer 2

n_hidden = R1.shape[1] + R2.shape[1]                 # 128 hidden neurons total
C = p.C

# --- key numbers ---
mean_rate = (R1.mean() + R2.mean()) / 2              # avg fraction of cycles a neuron fires
# spikes per decision = (sum of firing rates over all hidden neurons) * C cycles
spikes_per_decision = (R1.sum(axis=1) + R2.sum(axis=1)).mean() * C
# an ANN "activates" every hidden neuron every pass:
ann_activations = n_hidden                           # 128 dense activations per decision

per_neuron_rate = np.concatenate([R1.mean(axis=0), R2.mean(axis=0)])  # (128,)
dead = np.mean(per_neuron_rate < 0.01)               # fraction essentially silent
saturated = np.mean(per_neuron_rate > 0.99)          # fraction essentially always-on

print(f"states analyzed:            {len(states)}")
print(f"mean firing rate:           {mean_rate:.1%}  (fraction of cycles a neuron fires)")
print(f"spikes per decision:        {spikes_per_decision:.1f}  spikes")
print(f"ANN dense activations:      {ann_activations}  (every neuron, every pass)")
print(f"  -> SNN emits ~{spikes_per_decision / (ann_activations * C):.1%} of the "
      f"activity of a dense net over its {C} cycles")
print(f"dead neurons (<1% firing):  {dead:.1%}")
print(f"saturated neurons (>99%):   {saturated:.1%}")

# --- histogram: distribution of per-neuron average firing rate ---
plt.figure(figsize=(7, 5))
plt.hist(per_neuron_rate, bins=20, color="tab:purple", edgecolor="k")
plt.axvline(mean_rate, color="red", ls="--", label=f"mean {mean_rate:.1%}")
plt.xlabel("per-neuron average firing rate")
plt.ylabel("number of hidden neurons")
plt.title("firing-rate distribution across the 128 hidden neurons\n"
          "(healthy = a spread in the 0.1–0.4 range, few dead/saturated)")
plt.legend()
plt.tight_layout(); plt.savefig("firing_stats.png", dpi=120); plt.close()
print("saved firing_stats.png")

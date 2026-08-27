"""What do the spikes encode?  Measure each hidden neuron's firing RATE as a
function of the input, to see which neurons respond to which features."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from networks import SNNPolicy
from drone_env import DroneNavEnv

FEATURES = ["ray0", "ray1", "ray2", "ray3", "ray4", "ray5", "ray6", "ray7",
            "goal_dx", "goal_dy"]     # the 10 observation dimensions, in order

p = SNNPolicy()
p.load_state_dict(torch.load("snn_policy_best.pt"))
p.eval()


@torch.no_grad()
def layer_rates(obs):
    """Run the SNN forward by hand and return per-neuron firing rate (spikes/C)
    for each hidden layer. Same loop as SNNPolicy.forward, but we keep the spikes."""
    x = torch.as_tensor(obs, dtype=torch.float32)
    mem1 = p.lif1.init_leaky()
    mem2 = p.lif2.init_leaky()
    s1 = torch.zeros(p.fc1.out_features)
    s2 = torch.zeros(p.fc2.out_features)
    for _ in range(p.C):
        spk1, mem1 = p.lif1(p.fc1(x), mem1); s1 += spk1
        spk2, mem2 = p.lif2(p.fc2(spk1), mem2); s2 += spk2
    return (s1 / p.C).numpy(), (s2 / p.C).numpy()


# --- collect a big set of REAL states the drone actually visits ---
def collect_states(n_seeds=40):
    env = DroneNavEnv()
    states = []
    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            states.append(obs.copy())
            a, _ = p.act(torch.from_numpy(obs), greedy=False, temperature=0.3)
            obs, _, term, trunc, _ = env.step(a.item())
            done = term or trunc
    return np.array(states)                       # (N, 10)


print("collecting states...")
S = collect_states()
print(f"{len(S)} states collected")

# firing rate of every layer-1 neuron at every state -> (N, 64)
R1 = np.array([layer_rates(s)[0] for s in S])

# --- Part A: correlation heatmap (neuron x input-feature) ---
# corr[n, f] = correlation between neuron n's firing rate and input feature f
corr = np.zeros((R1.shape[1], S.shape[1]))
for n in range(R1.shape[1]):
    for f in range(S.shape[1]):
        if R1[:, n].std() > 1e-6:
            corr[n, f] = np.corrcoef(R1[:, n], S[:, f])[0, 1]

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_ylabel("hidden neuron (layer 1)")
ax.set_title("what each spiking neuron encodes\n(firing-rate vs input correlation)")
plt.colorbar(im, ax=ax, label="correlation")
plt.tight_layout(); plt.savefig("spike_encoding_heatmap.png", dpi=120); plt.close()
print("saved spike_encoding_heatmap.png")

# --- Part B: tuning curves for the neurons most tuned to one chosen feature ---
FEAT = 0                                          # which input to sweep (0 = ray0, forward)
top = np.argsort(-np.abs(corr[:, FEAT]))[:4]      # 4 neurons most correlated with it

sweep = np.linspace(0, 1, 25)
base = np.full(10, 0.5, dtype=np.float32)         # hold other inputs at mid-range
fig, ax = plt.subplots(figsize=(7, 5))
for n in top:
    rates = []
    for v in sweep:
        obs = base.copy(); obs[FEAT] = v
        rates.append(layer_rates(obs)[0][n])
    ax.plot(sweep, rates, marker="o", label=f"neuron {n} (corr {corr[n, FEAT]:+.2f})")
ax.set_xlabel(f"input: {FEATURES[FEAT]}"); ax.set_ylabel("firing rate")
ax.set_title(f"tuning curves: firing rate vs {FEATURES[FEAT]}")
ax.legend()
plt.tight_layout(); plt.savefig("spike_encoding_tuning.png", dpi=120); plt.close()
print("saved spike_encoding_tuning.png")

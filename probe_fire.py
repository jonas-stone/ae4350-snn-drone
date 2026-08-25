import torch
from drone_env import DroneNavEnv
from networks import SNNPolicy

env = DroneNavEnv()
obs, _ = env.reset(seed=0)
p = SNNPolicy(threshold=0.3) # instantiate a policy

# re-run the forward loop by hand, but tally spikes per layer
x = torch.from_numpy(obs)
mem1 = p.lif1.init_leaky(); mem2 = p.lif2.init_leaky()
f1 = f2 = 0
for _ in range(p.C):
    cur1 = p.fc1(x);            spk1, mem1 = p.lif1(cur1, mem1); f1 += spk1.sum().item()
    cur2 = p.fc2(spk1);         spk2, mem2 = p.lif2(cur2, mem2); f2 += spk2.sum().item()

print(f"layer1 firing rate: {f1/(p.C*64):.2%}")   # spikes / (cycles * neurons)
print(f"layer2 firing rate: {f2/(p.C*64):.2%}")
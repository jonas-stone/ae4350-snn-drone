import torch
import torch.nn as nn
import snntorch as snn

fc1 = nn.Linear(10, 64)                 # turns 10 observations -> 64 currents
lif1 = snn.Leaky(beta=0.9, threshold=1.0)
mem1 = lif1.init_leaky()


fc2 = nn.Linear(64, 64)
lif2 = snn.Leaky(beta=0.9, threshold=1.0)
mem2 = lif2.init_leaky()

fc3 = nn.Linear(64, 4)

out_sum = torch.zeros(4)

obs = torch.rand(10)                     # fake observation: 10 numbers (stand-in for rays+goal)

for cycle in range(8):
    cur1 = fc1(obs)                      # 10 obs -> 64 currents  (SAME obs every cycle)
    spk1, mem1 = lif1(cur1, mem1)        # feed those currents into the 64 LIF neurons
    cur2 = fc2(spk1)                    
    spk2, mem2 = lif2(cur2, mem2)
    out = fc3(spk2)
    out_sum = out_sum + out

    print(f"cycle {cycle+1}: fired {spk2.sum().item():.0f} of 64")

print(f"final output (logits): {out_sum / 8}")
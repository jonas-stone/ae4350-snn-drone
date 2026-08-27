import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=10, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),   # 10 -> 64
            nn.Tanh(),                    # nonlinearity
            nn.Linear(hidden, hidden),    # 64 -> 64
            nn.Tanh(),
            nn.Linear(hidden, n_actions), # 64 -> 4  (one score per action)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs, greedy=False, temperature=1.0):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits / temperature)  # T<1 sharpens
        if greedy: # just takes the highest prob. --> used for eval
            action = torch.argmax(logits, dim=-1)
        else: # this is stochastic and used for training
            action = dist.sample() # sample takes an action according to the probabilities of each option
        return action, dist.log_prob(action)

class ValueNet(nn.Module):
    def __init__(self, obs_dim=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),        # 10 -> 64 -> 64 -> 1  (a single value)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # drop trailing dim: (...,1) -> (...) scalar per state

import snntorch as snn
from snntorch import surrogate

class SNNPolicy(nn.Module):
    def __init__(self, obs_dim=10, n_actions=4, hidden=64, C=8, beta=0.9, threshold=0.3):
        super().__init__()
        self.C = C                          # number of integration cycles
        spike_grad = surrogate.atan()       # the fake-gradient trick
        
        # SAME skeleton as the MLP: three Linear layers, 10->64->64->4.
        # But between them, instead of nn.Tanh(), we put snn.Leaky (the LIF neuron).
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        # 1. Each LIF neuron has memory (V). Reset it at the start of a forward pass:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # 2. An accumulator for the output over all cycles (decoding = summing):
        out_sum = torch.zeros(self.fc3.out_features)                       # zeros, shape = (n_actions,)

        # 3. Run C cycles. Each cycle: inject the SAME obs (current injection),
        #    push it through the layers, let the LIF neurons integrate & maybe fire.
        for _ in range(self.C):
            cur1 = self.fc1(x)              # weighted sum -> current into layer 1
            spk1, mem1 = self.lif1(cur1, mem1)   # LIF: returns (spikes, new membrane)
            cur2 = self.fc2(spk1)                      # spikes from layer 1 -> current into layer 2
            spk2, mem2 = self.lif2(cur2, mem2)                # layer 2 LIF
            out  = self.fc3(spk2)           # read out this cycle's contribution
            out_sum = out_sum + out         # accumulate

        # 4. Decode: average over cycles -> the 4 logits
        return out_sum / self.C

    def act(self, obs, greedy=False, temperature=1.0):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits / temperature)  # T<1 sharpens
        if greedy: # just takes the highest prob. --> used for eval
            action = torch.argmax(logits, dim=-1)
        else: # this is stochastic and used for training
            action = dist.sample() # sample takes an action according to the probabilities of each option
        return action, dist.log_prob(action)
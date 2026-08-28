import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=10, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs, greedy=False, temperature=1.0):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits / temperature)   # t<1 sharpens
        action = torch.argmax(logits, dim=-1) if greedy else dist.sample()    # greedy=eval, sample=training
        return action, dist.log_prob(action)


class ValueNet(nn.Module):
    def __init__(self, obs_dim=10, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


import snntorch as snn
from snntorch import surrogate


class SNNPolicy(nn.Module):
    """same architecture as the mlp, but with leaky integrate-and-fire neurons."""
    def __init__(self, obs_dim=10, n_actions=4, hidden=64, C=8, beta=0.9, threshold=0.3):
        super().__init__()
        self.C = C                          # integration cycles per decision
        spike_grad = surrogate.atan()       # surrogate gradient (backprop through the spike)
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros(self.fc3.out_features)
        for _ in range(self.C):             # inject the same obs every cycle, accumulate the output
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            out_sum = out_sum + self.fc3(spk2)
        return out_sum / self.C             # spike-rate decoding -> logits

    def act(self, obs, greedy=False, temperature=1.0):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits / temperature)
        action = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action)

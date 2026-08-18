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

    def act(self, obs, greedy=False):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy: # just takes the highest prob. --> used for eval
            action = torch.argmax(logits, dim=-1)
        else: # this is stochastic and used for training
            action = dist.sample() # sample takes an action according to the probabilities of each option
        return action, dist.log_prob(action)
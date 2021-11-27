import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

class RL(nn.Module):

    def __init__(self, hidden_dim):
        super(RL, self).__init__()

        self.fc_rl = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        out = torch.sigmoid(self.fc_rl(x))
        m = Bernoulli(out)
        self.action = m.sample()
        self.logpro = torch.exp(m.log_prob(self.action))
        self.entrop = m.entropy()

        return x * self.action

    def cal_loss(self,reward,predictions,log_softmax, entropy):

        #baseline
        reward_mean = torch.mean(reward, dim=0)
        reward_baseline = reward - reward_mean - predictions.squeeze()
        loss = - torch.mean(torch.mean(reward_baseline.unsqueeze(1) * log_softmax + entropy,dim=1), dim=0)

        return loss

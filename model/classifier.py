import torch.nn as nn
import torch
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 11)

    def forward(self, x):

        x = self.fc(x)
        return torch.sigmoid(x)
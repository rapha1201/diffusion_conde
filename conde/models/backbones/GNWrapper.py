import torch
import torch.nn as nn

class GNWrapper(nn.Module):
    def __init__(self, C, groups=4):
        super().__init__()
        self.act = nn.GroupNorm(groups, C)
    def forward(self, x):
        return self.act(x)
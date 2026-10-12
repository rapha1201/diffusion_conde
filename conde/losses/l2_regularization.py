import torch
import torch.nn as nn

class L2Reg(nn.Module):
    def __init__(self, d=2):
        super(L2Reg, self).__init__()
        self.d=d
    def forward(self,z):
        return (z ** self.d).sum(dim=1).mean()
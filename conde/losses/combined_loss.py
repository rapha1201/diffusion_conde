import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            weights = [1.0] * len(losses)
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, *args, **kwargs):
        total = 0.0
        for w, loss_fn in zip(self.weights, self.losses):
            total = total + w * loss_fn(*args, **kwargs)
        return total
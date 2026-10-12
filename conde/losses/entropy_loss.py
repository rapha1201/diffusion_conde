import torch
import torch.nn as nn

class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits):
        probs = torch.sigmoid(logits)
        entropy = -(probs * torch.log(probs + self.eps) +
                    (1 - probs) * torch.log(1 - probs + self.eps))
        return entropy.mean()
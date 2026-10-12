import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalityLoss(nn.Module):
    """
    Penalizes non-orthogonality between two tensors.
    By default, assumes inputs are [B, D] and computes batchwise dot products.
    """
    def __init__(self, reduction: str = "mean", normalize: bool = True, sparse=False):
        """
        Args:
            reduction: 'mean' | 'sum' | 'none'
            normalize: if True, vectors are L2-normalized before dot product
        """
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize
        self.sparse = sparse

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        
        # dot products per sample
        if self.sparse:
            m = x*y
            m = torch.abs(m)
            loss = m.sum(dim=-1)
        else:
            dots = torch.sum(x * y, dim=-1)  
            loss = dots.pow(2)  # penalize squared correlation
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ExclusivityLoss(nn.Module):
    """
    Encourages each embedding (b, g, s, c) to occupy disjoint subsets
    of dimensions. Zero when dimensions are used by only one head.
    """

    def forward(self, w,x):


        # absolute values to measure activation magnitude
        V = torch.stack([w.abs(), x.abs()], dim=1)   # [B, 4, D]

        # sum of magnitudes per-token, per-dimension
        sum_v = V.sum(dim=1)                      # [B, D]
        sum_sq = (V ** 2).sum(dim=1)              # [B, D]

        # exclusivity penalty per dimension
        penalty = sum_v**2 - sum_sq               # [B, D]

        # mean over dimensions & batch
        return penalty.mean()

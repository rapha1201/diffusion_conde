import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, distance="l2"):
        """
        margin: float
            The enforced gap between positive and negative pairs.
        distance: str
            'l2' for Euclidean distance, 'cosine' for 1 - cosine similarity.
        """
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        anchor   = F.normalize(anchor,  dim=-1, eps=1e-6)
        positive = F.normalize(positive,dim=-1, eps=1e-6)
        negative = F.normalize(negative,dim=-1, eps=1e-6)
        if self.distance == "l2":
            d_ap = F.pairwise_distance(anchor, positive, p=2)
            d_an = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance == "cosine":
            d_ap = 1 - F.cosine_similarity(anchor, positive)
            d_an = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance}")

        losses = F.relu(d_ap - d_an + self.margin)
        return losses.mean()
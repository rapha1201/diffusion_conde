import torch
import torch.nn as nn
import torch.nn.functional as F

class NtXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1, emb2, emb3=None):

        features = torch.stack((emb1, emb2), dim=0)
        features = features.permute(1, 0, 2).reshape(emb1.shape[0], 2, features.shape[-1])
        features = features.reshape(-1, features.shape[-1])
        
        # Normalize features for cosine similarity
        features = F.normalize(features, dim=-1)

        # Compute similarity matrix using matrix multiplication
        sim_matrix = features @ features.T  # [N, N], cosine similarity
        sim_matrix = sim_matrix / self.temperature

        # Remove self-similarity by masking diagonal
        diag_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(diag_mask, float("-inf"))

        # Create positive pair targets
        targets = torch.arange(features.shape[0], device=features.device)
        targets[0::2] += 1
        targets[1::2] -= 1

        return F.cross_entropy(sim_matrix, targets)
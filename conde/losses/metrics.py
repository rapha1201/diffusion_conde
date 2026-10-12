import torch
import torch.nn as nn

class ExplainedVariance:
    """
    Computes:
        EV_sem  = 1 - MSE(Z, S)      / Var(Z)
        EV_full = 1 - MSE(Z, S+R)    / Var(Z)

    Use:
        ev = ExplainedVariance()
        ev_sem, ev_full = ev(Z, S, R)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, Z, S, R):
        # Z: [B, D]
        # S: [B, D]
        # R: [B, D]

        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        varZ = (Z_centered ** 2).mean() + self.eps

        mse_sem  = ((Z - S)**2).mean()
        mse_full = ((Z - (S + R))**2).mean()

        EV_sem  = 1 - mse_sem  / varZ
        EV_full = 1 - mse_full / varZ

        return EV_sem.item(), EV_full.item()

class MRRMetric:
    """
    Computes MRR for (anchor, positive) retrieval.
    Use:
        mrr = MRRMetric()
        score = mrr(anchors, positives)
    """
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, A, P):
        # A, P: [B, D]
        A_norm = A / (A.norm(dim=1, keepdim=True) + self.eps)
        P_norm = P / (P.norm(dim=1, keepdim=True) + self.eps)

        # Similarity matrix: sim[i, j] = cos(A_i, P_j)
        sim = A_norm @ P_norm.T  # [B, B]

        # For each anchor i, the positive is sim[i, i]
        pos_scores = sim.diag().unsqueeze(1)    # [B, 1]

        # Rank positives by comparing with all scores for same anchor
        ranks = (sim >= pos_scores).sum(dim=1)

        # Reciprocal rank
        rr = 1.0 / ranks.float()

        # Mean Reciprocal Rank
        return rr.mean().item()

class DecoderGap:
    """
    Measures similarity between:
        Dec(Z)            == AE reconstruction
        Dec(S_pos + R)    == decomposition reconstruction

    By default uses MSE. If "similarity=True", returns (1 - MSE).
    """
    def __init__(self, similarity=False, eps=1e-8):
        self.similarity = similarity
        self.eps = eps

    def __call__(self, X_ae, X_sr):
        # both tensors: [B, C, H, W] or [B, T] for audio
        mse = ((X_ae - X_sr)**2).mean()

        if self.similarity:
            # return similarity score in [0, 1]
            return max(0.0, 1.0 - mse.item())
        else:
            # return raw MSE gap
            return mse.item()
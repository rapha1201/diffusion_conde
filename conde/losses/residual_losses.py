import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_xy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expect x,y as (B, D). Also accept (B,) and treat as (B,1).
    y is assumed detached by caller, but we do not enforce it here.
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"x and y must be 1D or 2D tensors. Got x={tuple(x.shape)}, y={tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Batch sizes must match. Got x={x.shape[0]} vs y={y.shape[0]}")
    if x.dtype != y.dtype:
        # Keep x's dtype; cast y for safe ops (y is detached anyway).
        y = y.to(dtype=x.dtype)
    return x, y


class CosineOrthogonalityLoss(nn.Module):
    """
    Per-sample cosine similarity squared between x and y, averaged over batch.
    Minimizing encourages x ⟂ y (weak, but very cheap).
      L = E_b [ cos(x_b, y_b)^2 ]
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _check_xy(x, y)
        x_n = F.normalize(x, p=2, dim=-1, eps=self.eps)
        y_n = F.normalize(y, p=2, dim=-1, eps=self.eps)
        cos = (x_n * y_n).sum(dim=-1)  # (B,)
        return (cos ** 2).mean()


class CrossCovarianceLoss(nn.Module):
    """
    Cross-covariance Frobenius norm:
      x_c = x - mean(x)
      y_c = y - mean(y)
      Sigma_xy = (x_c^T y_c)/(B-1)
      L = ||Sigma_xy||_F^2

    Captures linear dependence across all dims. Stable and fast: O(BD^2) for matmul.
    """
    def __init__(self, eps: float = 1e-12, unbiased: bool = True):
        super().__init__()
        self.eps = eps
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _check_xy(x, y)
        b = x.shape[0]
        if b < 2:
            return x.new_tensor(0.0)

        x_c = x - x.mean(dim=0, keepdim=True)
        y_c = y - y.mean(dim=0, keepdim=True)

        denom = (b - 1) if self.unbiased else b
        denom = max(1, denom)

        sigma_xy = (x_c.T @ y_c) / (denom + self.eps)  # (Dx, Dy)
        return (sigma_xy ** 2).mean()


class CrossCorrelationLoss(nn.Module):
    """
    Barlow Twins–style cross-correlation penalty between x and y:
      x_z = (x - mean)/std
      y_z = (y - mean)/std
      C = (x_z^T y_z) / B
      L = ||C||_F^2

    Similar to CrossCovarianceLoss but explicitly standardizes dimensions.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _check_xy(x, y)
        b = x.shape[0]
        if b < 2:
            return x.new_tensor(0.0)

        x_mu = x.mean(dim=0, keepdim=True)
        y_mu = y.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
        y_std = y.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)

        x_z = (x - x_mu) / x_std
        y_z = (y - y_mu) / y_std

        c = (x_z.T @ y_z) / b  # (Dx, Dy)
        return (c ** 2).mean()


def _double_center(a: torch.Tensor) -> torch.Tensor:
    """
    Double-center a batchwise pairwise distance matrix a (B,B):
      A_ij <- A_ij - mean_row_i - mean_col_j + mean_all
    """
    mean_row = a.mean(dim=1, keepdim=True)
    mean_col = a.mean(dim=0, keepdim=True)
    mean_all = a.mean()
    return a - mean_row - mean_col + mean_all


class DistanceCorrelationLoss(nn.Module):
    """
    Distance correlation (dCor) loss. 0 iff independent (under mild conditions).
    O(B^2) memory/time due to pairwise distances.

    Steps:
      A = pairwise_dist(x), B = pairwise_dist(y)
      A~ = double_center(A), B~ = double_center(B)
      dCov^2 = mean(A~ * B~)
      dVar_x = mean(A~ * A~), dVar_y = mean(B~ * B~)
      dCor = dCov / sqrt(dVar_x * dVar_y)

    We minimize dCor^2 for stability.
    """
    def __init__(self, eps: float = 1e-12, p: float = 2.0):
        super().__init__()
        self.eps = eps
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _check_xy(x, y)
        b = x.shape[0]
        if b < 2:
            return x.new_tensor(0.0)

        # Pairwise distances
        A = torch.cdist(x, x, p=self.p)  # (B,B)
        B = torch.cdist(y, y, p=self.p)  # (B,B)

        A = _double_center(A)
        B = _double_center(B)

        dcov2 = (A * B).mean()
        dvarx = (A * A).mean().clamp_min(self.eps)
        dvary = (B * B).mean().clamp_min(self.eps)

        # Minimize squared distance correlation
        dcor2 = (dcov2 ** 2) / (dvarx * dvary)
        return dcor2


class AntiInfoNCELoss(nn.Module):
    """
    "Anti"-InfoNCE: penalize alignment of matching (x_b, y_b) relative to mismatched pairs.
    This is NOT a min-max adversary; it directly trains x to *not* be predictive of y.

    We compute logits = sim(x, y) / tau (B,B) and take the standard NCE loss,
    but we REVERSE the objective: we *maximize* the NCE loss w.r.t. x.

    Implementation: return (-nce) so minimizing this makes NCE larger (worse alignment).

    IMPORTANT: This can encourage collapse (e.g., shrinking x norms). Consider pairing
    with a variance/energy regularizer on x if needed.
    """
    def __init__(self, tau: float = 0.2, eps: float = 1e-8, use_cosine: bool = True):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.use_cosine = use_cosine

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _check_xy(x, y)
        b = x.shape[0]
        if b < 2:
            return x.new_tensor(0.0)

        if self.use_cosine:
            x_n = F.normalize(x, p=2, dim=-1, eps=self.eps)
            y_n = F.normalize(y, p=2, dim=-1, eps=self.eps)
            logits = (x_n @ y_n.T) / self.tau  # (B,B)
        else:
            logits = (x @ y.T) / self.tau

        targets = torch.arange(b, device=x.device)
        nce = F.cross_entropy(logits, targets)
        # Reverse objective: minimize (-nce) -> maximize nce (reduce dependence)
        return -nce


class VarianceFloorLoss(nn.Module):
    """
    Optional helper to prevent x (residual) from collapsing:
      L = mean_j ReLU(gamma - std(x[:,j]))^2

    Use with a small weight (e.g., 1e-3 to 1e-2) if AntiInfoNCE or strong decorrelation
    drives x toward zero-variance.
    """
    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # signature accepts (x,y) style; ignore y if provided
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        std = x.std(dim=0, unbiased=False) + self.eps
        return F.relu(self.gamma - std).pow(2).mean()

class IndependentDimensionalityLoss(nn.Module):

    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps=eps

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:

        x_n = F.normalize(x, p=2, dim=-1, eps=self.eps)
        y_n = F.normalize(y, p=2, dim=-1, eps=self.eps)
        return (x_n.pow(2) * y_n.pow(2)).mean().sqrt()

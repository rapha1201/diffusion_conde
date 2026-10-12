import math
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, eps=1e-8):
    # x: [B,H,Dh]
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def mcl_winners(za, zp, zn, margin=0.2, beta=10.0):
    """
    Compute per-head triplet loss L[b,h] and return hard winner indices per sample.
    No grads flow through the routing.
    """
    A, P, N = map(l2norm, (za, zp, zn))           # [B,H,Dh]
    d_pos = (A - P).pow(2).sum(-1).sqrt()         # [B,H]
    d_neg = (A - N).pow(2).sum(-1).sqrt()         # [B,H]
    L = F.relu(margin + d_pos - d_neg)            # [B,H]
    # hard winner (argmin)
    winners = torch.argmin(L, dim=1)              # [B]
    return winners, L                              # return L if you also need usage stats

def swap_winner_heads(za, zp, winners):
    """
    Replace the winner head of anchor with the positive’s head.
    za, zp: [B,H,Dh], winners: [B]
    Returns a new tensor za_swap with the swap applied.
    """
    za_swap = za.clone()
    b_idx = torch.arange(za.size(0), device=za.device)
    za_swap[b_idx, winners, :] = zp[b_idx, winners, :]
    return za_swap

class MCLTripletLoss(nn.Module):
    """
    Multiple-Choice Learning (MCL) over heads with a scheduled beta (routing sharpness).

    Inputs to forward:
      za, zp, zn : [B, H, Dh]  (anchor/positive/negative per head)

    Returns:
      total_loss : scalar
      stats      : dict with L_mcl, L_usage, beta, usage (H,), w (B,H) optionally for logging

    Notes:
      - Routing weights w are computed from per-head losses and (optionally) detached.
      - usage_weight balances head load via entropy; set >0 to discourage collapse.
      - hard_after_step switches to top-1 routing after that step (optional).
    """

    def __init__(self,
                 margin: float = 0.5,
                 beta_start: float = 2.0,
                 beta_end: float = 20.0,
                 total_steps: int = 100_000,
                 schedule: str = "cosine",   # "linear" or "cosine"
                 detach_routing: bool = True,
                 usage_weight: float = 0.0,  # lambda for usage entropy
                 hard_after_step: int | None = None,
                 normalize: bool = True,
                 eps: float = 1e-8,
                 return_weights: bool = False):
        super().__init__()
        assert schedule in ("linear", "cosine")
        self.margin = margin
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_steps = int(max(1, total_steps))
        self.schedule = schedule
        self.detach_routing = detach_routing
        self.usage_weight = float(usage_weight)
        self.hard_after_step = hard_after_step
        self.normalize = normalize
        self.eps = eps
        self.return_weights = return_weights

        # Internal step counter (can be driven automatically or overridden per call)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long), persistent=True)

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)

    def current_beta(self, step: int | None = None) -> float:
        s = int(self._step.item() if step is None else step)
        t = min(max(s / self.total_steps, 0.0), 1.0)
        if self.schedule == "linear":
            beta = self.beta_start + (self.beta_end - self.beta_start) * t
        else:  # cosine
            beta = self.beta_start + (self.beta_end - self.beta_start) * (1 - math.cos(math.pi * t)) * 0.5
        return float(beta)

    def _routing_context(self):
        # Detach routing by default; otherwise allow gradients through w (rarely desirable)
        return torch.no_grad() if self.detach_routing else contextlib.nullcontext()

    def forward(self,
                za: torch.Tensor,  # [B,H,Dh]
                zp: torch.Tensor,  # [B,H,Dh]
                zn: torch.Tensor,  # [B,H,Dh]
                step: int | None = None) -> tuple[torch.Tensor, dict]:
        assert za.dim() == zp.dim() == zn.dim() == 3, "Expect [B,H,Dh] tensors"

        if self.normalize:
            za = self._l2norm(za)
            zp = self._l2norm(zp)
            zn = self._l2norm(zn)

        # Distances per head & sample
        d_pos = (za - zp).pow(2).sum(-1).sqrt()   # [B,H]
        d_neg = (za - zn).pow(2).sum(-1).sqrt()   # [B,H]
        L = F.relu(self.margin + d_pos - d_neg)   # [B,H], per-head triplet

        # Beta (schedule)
        beta = self.current_beta(step)

        # Routing weights over heads
        with self._routing_context():
            if self.hard_after_step is not None and (step is not None and step >= self.hard_after_step
                                                     or step is None and self._step.item() >= self.hard_after_step):
                # Top-1 hard gating (winner-take-all)
                idx = torch.argmin(L, dim=1)               # [B]
                w = torch.zeros_like(L)
                w[torch.arange(L.size(0), device=L.device), idx] = 1.0
            else:
                w = torch.softmax(-beta * L, dim=1)        # [B,H]

        # MCL aggregation
        L_mcl = (w * L).sum(dim=1).mean()                 # scalar

        # Usage entropy (load balancing)
        usage = w.mean(dim=0)                             # [H]
        L_usage = -(usage * (usage + self.eps).log()).sum()

        total = L_mcl + self.usage_weight * L_usage

        # Auto-increment internal step counter if caller didn't pass one
        if step is None:
            self._step += 1

        stats = {
            "L_mcl": L_mcl.detach(),
            "L_usage": L_usage.detach(),
            "beta": torch.tensor(beta, device=L.device),
            "usage": usage.detach(),
        }
        if self.return_weights:
            stats["w"] = w.detach()

        return total

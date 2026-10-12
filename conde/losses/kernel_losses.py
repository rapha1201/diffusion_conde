import torch
import torch.nn as nn

class RBFKernelBase:
    def __init__(self, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0), batch=None):
        """
        Base class providing:
            • automatic sigma selection
            • RBF multi-sigma kernel computation
        Child classes implement their own: __call__(x, y)
        """
        if batch is not None:
            self.sigmas = self._auto_sigmas(batch)
        else:
            self.sigmas = sigmas

    def _auto_sigmas(self, batch):
        """
        Median distance heuristic + geometric ladder, identical to your version.
        Only affects the sigma bandwidths; losses defined in child classes.
        """
        with torch.no_grad():
            x = batch
            if x.size(0) > 512:
                x = x[:512]

            diffs = x[:, None] - x[None, :]
            dist = (diffs.pow(2).sum(-1)).sqrt()
            m = dist.median().item()

        return [m/8, m/4, m/2, m, 2*m, 4*m]

    def rbf_kernel(self, x, y):
        """
        Multi-scale RBF kernel sum:  K = Σ exp(-||x-y||² / 2σ²)
        """
        x2 = (x**2).sum(1, keepdim=True)
        y2 = (y**2).sum(1, keepdim=True)
        dist = x2 - 2 * (x @ y.t()) + y2.t()

        K = 0.0
        for s in self.sigmas:
            K = K + torch.exp(-dist / (2 * s * s))
        K = K/len(self.sigmas)
        return K



class MMD_RBF(RBFKernelBase):
    def __call__(self, x, y):
        """
        Biased MMD estimator:
            MMD = E[Kxx] + E[Kyy] - 2 E[Kxy]
        """
        Kxx = self.rbf_kernel(x, x)
        Kyy = self.rbf_kernel(y, y)
        Kxy = self.rbf_kernel(x, y)

        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()



class HSIC_RBF(torch.nn.Module):
    def __init__(self, sigma: float | None = None, eps: float = 1e-12, use_median_heuristic: bool = True):
        super().__init__()
        self.sigma = sigma
        self.eps = eps
        self.use_median_heuristic = use_median_heuristic

    def _pairwise_sq_dists(self, x: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        d2 = x2 + x2.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))
        return d2.clamp_min(0.0)

    def _rbf(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        d2 = self._pairwise_sq_dists(x)
        gamma = 1.0 / (2.0 * (sigma * sigma).clamp_min(self.eps))
        # exp of negative values is safe; clamp argument if you want to avoid underflow-to-zero
        return torch.exp(-gamma * d2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if n < 4:
            return x.new_zeros(())  # unbiased estimator undefined


        # choose sigma
        if self.sigma is None and self.use_median_heuristic:
            with torch.no_grad():
                d2x = self._pairwise_sq_dists(x)
                d2y = self._pairwise_sq_dists(y)
                mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
                sig_x = torch.sqrt(0.5 * d2x[mask].median().clamp_min(self.eps))
                sig_y = torch.sqrt(0.5 * d2y[mask].median().clamp_min(self.eps))
                sigma = torch.sqrt(sig_x * sig_y).clamp_min(self.eps)  # single shared sigma
        else:
            sigma = torch.tensor(float(self.sigma), device=x.device, dtype=x.dtype).clamp_min(self.eps)

        K = self._rbf(x, sigma)
        L = self._rbf(y, sigma)

        # zero diagonals without in-place fill
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        K = K * mask
        L = L * mask

        # do accumulation in higher precision (helps)
        Kd = K.double()
        Ld = L.double()

        K_sum = Kd.sum()
        L_sum = Ld.sum()
        K_row_sum = Kd.sum(dim=1)
        L_row_sum = Ld.sum(dim=1)

        term1 = (Kd * Ld).sum()
        term2 = (K_row_sum * L_row_sum).sum()

        hsic = (
            term1
            - (2.0 * term2) / (n - 2)
            + (K_sum * L_sum) / ((n - 1) * (n - 2))
        )

        out = hsic / (n * (n - 3))
        return out.to(dtype=x.dtype)


class HSIC_RBF_Multi(nn.Module):
    """
    Stable HSIC for training:
      - standardizes x,y per-dim
      - uses multi-sigma RBF mixture (median heuristic * ladder)
      - uses centered (biased) HSIC: tr(Kc Lc)/(n-1)^2
    """
    def __init__(
        self,
        ladder=(1/8, 1/4, 1/2, 1.0, 2.0, 4.0),
        max_sigma_samples=512,
        eps=1e-12,
        standardize=True,
        detach_sigma=True,
    ):
        super().__init__()
        self.ladder = ladder
        self.max_sigma_samples = max_sigma_samples
        self.eps = eps
        self.standardize = standardize
        self.detach_sigma = detach_sigma

    def _pairwise_sq_dists(self, x: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        d2 = x2 + x2.t() - 2.0 * (x @ x.t())
        return d2.clamp_min(0.0)

    def _median_sigma(self, x: torch.Tensor) -> torch.Tensor:
        # median of pairwise distances (excluding diagonal)
        n = x.size(0)
        d2 = self._pairwise_sq_dists(x)
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        med = d2[mask].median().clamp_min(self.eps)
        # match common convention: sigma^2 = 0.5 * median(d^2)
        return torch.sqrt(0.5 * med).clamp_min(self.eps)

    def _rbf_mix(self, x: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
        d2 = self._pairwise_sq_dists(x)
        K = 0.0
        for m in self.ladder:
            s = sigma0 * float(m)
            gamma = 1.0 / (2.0 * (s * s).clamp_min(self.eps))
            K = K + torch.exp(-gamma * d2)
        return K / len(self.ladder)

    def _center(self, K: torch.Tensor) -> torch.Tensor:
        n = K.size(0)
        one_n = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1: x = x.unsqueeze(-1)
        if y.dim() == 1: y = y.unsqueeze(-1)
        n = x.size(0)
        if n < 4:
            return x.new_zeros(())

        if self.standardize:
            x = (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
            y = (y - y.mean(0, keepdim=True)) / y.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)

        # sigma estimation on subset (speed + stability)
        xs = x
        ys = y
        if n > self.max_sigma_samples:
            xs = x[:self.max_sigma_samples]
            ys = y[:self.max_sigma_samples]

        sigma_x = self._median_sigma(xs.detach() if self.detach_sigma else xs)
        sigma_y = self._median_sigma(ys.detach() if self.detach_sigma else ys)
        sigma0 = torch.sqrt(sigma_x * sigma_y).clamp_min(self.eps)

        K = self._rbf_mix(x, sigma0)
        L = self._rbf_mix(y, sigma0)

        Kc = self._center(K)
        Lc = self._center(L)

        # biased HSIC (stable)
        hsic = (Kc * Lc).sum() / ((n - 1) ** 2)
        return hsic


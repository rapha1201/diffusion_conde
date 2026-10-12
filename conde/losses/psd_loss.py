import torch
import torch.nn as nn
import torch.nn.functional as F

class PSDFrequencyLoss(nn.Module):
    """
    Power Spectral Density (PSD) loss module.
    Computes the radial PSD of both the original and reconstructed images
    and returns the MSE between their log-scaled PSDs.
    """

    def __init__(self, nbins: int = None, eps: float = 1e-6):
        """
        Args:
            nbins (int): Number of radial frequency bins. If None, uses min(H, W)//2.
            eps (float): Small constant for numerical stability in log.
        """
        super().__init__()
        self.nbins = nbins
        self.eps = eps

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Original images, shape (B, 1, H, W).
            x_hat (torch.Tensor): Reconstructed images, same shape as x.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        psd_x     = self._radial_psd(x)
        psd_hat   = self._radial_psd(x_hat)

        # Optional log scaling to emphasize higher frequencies
        log_x   = torch.log1p(psd_x + self.eps)
        log_hat = torch.log1p(psd_hat + self.eps)

        return F.mse_loss(log_hat, log_x)

    def _radial_psd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute radial PSD for a batch of images.
        
        Args:
            x (torch.Tensor): Input batch, shape (B, 1, H, W).
        
        Returns:
            torch.Tensor: Radial PSD, shape (B, nbins).
        """
        B, _, H, W = x.shape

        # Compute 2D FFT and power spectrum
        X = torch.fft.rfft2(x, norm='ortho')  # (B, 1, H, W//2+1), complex
        P = (X.abs() ** 2) / (H * W)          # normalized periodogram

        # Create radius map
        device = x.device
        ys = torch.arange(H, device=device) - H // 2
        xs = torch.arange(W // 2 + 1, device=device) - (W // 2)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        r = torch.sqrt(yy**2 + xx**2).flatten()  # (H*(W//2+1),)

        # Determine number of bins
        if self.nbins is None:
            nbins = min(H, W) // 2
        else:
            nbins = self.nbins

        edges = torch.linspace(0, r.max(), nbins + 1, device=device)
        psd_radial = torch.zeros((B, nbins), device=device)

        # Bin and average
        for i in range(nbins):
            mask = (r >= edges[i]) & (r < edges[i+1])
            if mask.any():
                vals = P[:, 0].flatten(1)  # (B, H*(W//2+1))
                psd_radial[:, i] = vals[:, mask].mean(dim=1)

        return psd_radial

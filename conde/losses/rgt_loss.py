import torch
import torch.nn as nn
import torch.nn.functional as F


class RGTLoss(nn.Module):
    """
    Loss for RGT prediction:
        total = w_ssim * SSIM_loss
              + w_mae  * MAE
              + w_grad * gradient_MAE
              + w_dgrad * second_gradient_MAE

    Assumes pred and target have shape:
        [B, 1, H, W] or [B, H, W]
    """

    def __init__(
        self,
        w_ssim=1.0,
        w_mae=0.2,
        w_grad=0.15,
        w_dgrad=0.1,
        ssim_window_size=11,
        ssim_sigma=1.5,
        data_range=1.0,
        eps=1e-8,
    ):
        super().__init__()

        self.w_ssim = w_ssim
        self.w_mae = w_mae
        self.w_grad = w_grad
        self.w_dgrad = w_dgrad
        self.data_range = data_range
        self.eps = eps

        self.ssim_window_size = ssim_window_size
        self.ssim_sigma = ssim_sigma

        window = self._make_gaussian_window(ssim_window_size, ssim_sigma)
        self.register_buffer("window", window)

    def _make_gaussian_window(self, size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        window_2d = g[:, None] @ g[None, :]
        window_2d = window_2d[None, None, :, :]
        return window_2d

    def _ssim_loss(self, pred, target):
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
    
        padding = self.ssim_window_size // 2
        window = self.window.to(dtype=pred.dtype, device=pred.device)
    
        pred_pad = F.pad(pred, (padding, padding, padding, padding), mode="reflect")
        target_pad = F.pad(target, (padding, padding, padding, padding), mode="reflect")
    
        mu_x = F.conv2d(pred_pad, window, padding=0)
        mu_y = F.conv2d(target_pad, window, padding=0)
    
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
    
        sigma_x2 = F.conv2d(pred_pad * pred_pad, window, padding=0) - mu_x2
        sigma_y2 = F.conv2d(target_pad * target_pad, window, padding=0) - mu_y2
        sigma_xy = F.conv2d(pred_pad * target_pad, window, padding=0) - mu_xy
    
    
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    
        ssim_map = numerator / (denominator + self.eps)
    
    
        return 1.0 - ssim_map.mean()

    def _gradients(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def _second_gradients(self, x):
        dx, dy = self._gradients(x)

        dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]

        # Mixed derivatives, useful for local curvature consistency.
        dxy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
        dyx = dy[:, :, :, 1:] - dy[:, :, :, :-1]

        return dxx, dyy, dxy, dyx

    def _grad_loss(self, pred, target):
        pred_dx, pred_dy = self._gradients(pred)
        target_dx, target_dy = self._gradients(target)

        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y

    def _double_grad_loss(self, pred, target):
        pred_dxx, pred_dyy, pred_dxy, pred_dyx = self._second_gradients(pred)
        target_dxx, target_dyy, target_dxy, target_dyx = self._second_gradients(target)

        loss = (
            F.l1_loss(pred_dxx, target_dxx)
            + F.l1_loss(pred_dyy, target_dyy)
            + F.l1_loss(pred_dxy, target_dxy)
            + F.l1_loss(pred_dyx, target_dyx)
        )

        return loss

    def forward(self, pred, target, return_dict=False):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        mae = F.l1_loss(pred, target)
        ssim = self._ssim_loss(pred, target)
        grad = self._grad_loss(pred, target)
        dgrad = self._double_grad_loss(pred, target)

        total = (
            self.w_ssim * ssim
            + self.w_mae * mae
            + self.w_grad * grad
            + self.w_dgrad * dgrad
        )

        if return_dict:
            return {
                "loss": total,
                "ssim": ssim.detach(),
                "mae": mae.detach(),
                "grad": grad.detach(),
                "double_grad": dgrad.detach(),
            }

        return total
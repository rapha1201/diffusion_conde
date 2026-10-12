import torch
import torch.nn as nn

class GradientLoss(nn.Module):
    """
    Gradient Difference Loss for 2D images.
    Computes the L^p difference of horizontal and vertical gradients
    between input and target.
    """
    def __init__(self, p: int = 1):
        """
        Args:
            p (int): the exponent for the norm (1 for L1, 2 for L2).
        """
        super(GradientLoss, self).__init__()
        self.p = p

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input  (Tensor): predicted images, shape (B, C, H, W)
            target (Tensor): ground-truth images, shape (B, C, H, W)
        Returns:
            loss (Tensor): scalar gradient loss
        """
        # horizontal gradients: differences along height (H)
        dx_input  = input[:, :, 1:, :] - input[:, :, :-1, :]
        dx_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        # vertical gradients: differences along width (W)
        dy_input  = input[:, :, :, 1:] - input[:, :, :, :-1]
        dy_target = target[:, :, :, 1:] - target[:, :, :, :-1]

        # compute L^p differences
        loss_x = torch.mean(torch.abs(dx_input - dx_target) ** self.p)
        loss_y = torch.mean(torch.abs(dy_input - dy_target) ** self.p)

        return loss_x + loss_y

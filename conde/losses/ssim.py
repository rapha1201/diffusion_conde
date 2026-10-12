import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class MSSSIMLoss(nn.Module):
    """(1 – MS-SSIM) as a loss term."""
    def __init__(self, data_range: float = 1.0, size_average: bool = True):
        super().__init__()
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ms_ssim returns a similarity score in [0,1]
        sim = ms_ssim(input, target,
                      data_range=self.data_range,
                      size_average=self.size_average,
                     weights=[1, 1, 1])
        return 1.0 - sim

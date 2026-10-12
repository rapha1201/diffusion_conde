from .ssim import MSSSIMLoss
from .charbonnier import CharbonnierLoss
from .fourier_loss import FourierLoss, DoubleFourierLoss
from .ntxentloss import NtXentLoss
from .triplet_loss import TripletLoss
from .psd_loss import PSDFrequencyLoss
from .grad_loss import GradientLoss
from .l2_regularization import L2Reg
from .mcl_triplet import MCLTripletLoss
from .orthogonality_loss import OrthogonalityLoss, ExclusivityLoss
from .entropy_loss import EntropyLoss
from .metrics import ExplainedVariance, MRRMetric, DecoderGap
from .kernel_losses import MMD_RBF, HSIC_RBF, HSIC_RBF_Multi
from .residual_losses import CosineOrthogonalityLoss, CrossCovarianceLoss, CrossCorrelationLoss, DistanceCorrelationLoss, AntiInfoNCELoss, VarianceFloorLoss, IndependentDimensionalityLoss
from .combined_loss import CombinedLoss
from .rgt_loss import RGTLoss

# Auto-register all classes manually imported
import inspect
import sys
import torch.nn as nn

_loss_registry = {'mae': nn.L1Loss, 'mse': nn.MSELoss, 'bce': nn.BCEWithLogitsLoss}

for name, obj in list(globals().items()):
    if inspect.isclass(obj):
        _loss_registry[name.lower()] = obj

def get_loss(name: str):
    return _loss_registry.get(name.lower())
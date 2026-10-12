import torchmetrics
import torch
from torchmetrics import JaccardIndex
from torchvision.transforms.functional import gaussian_blur
from scipy.ndimage import binary_dilation

def minmax_normalize(tensor: torch.tensor) -> torch.tensor:
    '''
        Performs Min-Max normalization on batched tensor.

        Args:
        -----
            - tensor (torch.tensor): a tensor with batch as it's first dimension.      
    '''
    
    batch_size, *other_axes = tensor.shape
    if batch_size == 1:
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            return torch.clamp(tensor, 0, 1)
    
        return (tensor - tensor.min())/(tensor.max() - tensor.min() + 1e-8)

    tensor = tensor.view(batch_size, -1)

    min_val, _ = torch.min(tensor, dim = 1, keepdim = True)
    max_val,_ = torch.max(tensor, dim = 1, keepdim = True)
    new_max = max_val + (max_val == min_val).float() # Avoiding division by 0 if all elements are equal
    tensor -= min_val
    tensor /= (new_max - min_val)
    tensor = tensor.view(batch_size, *other_axes)

    max_val = max_val.squeeze()
    s = torch.sum(tensor, list(range(1, len(other_axes) + 1)), keepdim = False) == 0

    for i in range(batch_size):
        tensor[i] += torch.clamp(s[i] * max_val[i], 0, 1) # Maintaining values in images where all elements are equal

    return tensor

def tol(tensor: torch.Tensor, sigma: int = 3) -> float:
    """
        Calculates the tolerance function as described in Ground Truth Uncertainty-Aware metrics
        for Machine Learning Applications on Seismic Image Interpretation Application to Faults and Horizon Extraction.

        Args:
        -----
            - tensor (torch.Tensor): Input tensor for tolerance calculation.
            - sigma (int, optional): Standard deviation parameter for the tolerance function. Default is 3.

        Returns:
        --------
            - float: The calculated tolerance value.

        Raises:
        -------
            - ValueError: If the input tensor is not a valid torch.Tensor.
    """

    return torch.exp((-tensor**2) / (2*sigma ** 2))

def batched_dilation(tensor: torch.tensor, kernel_size: int = 5) -> torch.tensor:
    '''
        Morphologically dialtes the input image.
        
        Args:
        -----
            - tensor (torch.tensor): a tensor with batch as it's first dimension.
            
            - kernel_size (int, optional): size of dilation kernel, defaults to 5.
                                           This must be an odd integer greater than 0.
    '''

    batch_size, *other_axes = tensor.shape
    kernel = torch.ones([kernel_size]*len(other_axes)).float()
    
    results = torch.empty(tensor.shape)
    for i in range(len(tensor)):
        results[i] = torch.as_tensor(binary_dilation(tensor[i], kernel))
        
    return results
    

class IoU:
    def __init__(self, **params):
        self.iou = JaccardIndex('binary', average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor):
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        iou = self.iou.to(device)
        res = iou(predicted, target)
        
        return res.item() 

class MeanIoU:
    def __init__(self, **params):
        self.iou = IoU(**params)
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:    
        return (self.iou(predicted, target) + self.iou(1 - predicted, 1 - target)) / 2

class GaussianFilter:
    def __init__(self, normalize: bool = True, kernel_size: int = 15, sigma : int = 5, **params):
        self.params = params
        self.normalize = normalize
        self.filter = gaussian_blur
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.squeeze(img, 1)
        filtered = self.filter(img.detach(), self.kernel_size, **self.params)
        
        if self.normalize:
            filtered = minmax_normalize(filtered)

        return filtered.unsqueeze(1)


class GaussianF1:
    def __init__(self, **params):
        self.params = params
        self.filter = GaussianFilter(**params)

    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        batch_size, *other_axes = predicted.shape
        axes = list(range(1, len(other_axes) + 1))
        
        target = target.detach()
        target_filtered = self.filter(target)
        
        predicted = predicted.detach()
        predicted_filtered = self.filter(predicted)

        N = (predicted_filtered*torch.abs(1-predicted)) + predicted
        FN = (torch.sum((target)*(1-N), dim=axes))
        TN = (torch.sum((1-target)*(1-N), dim=axes))

        P = (target_filtered*torch.abs(1-target)) + target
        TP = (torch.sum(P*predicted, dim=axes))
        FP = (torch.sum((1-P)*(predicted), dim=axes))

        f1 = (TP / (TP + 0.5*(FP + FN)))
        f1[f1 != f1] = 0 # Replaces NaN with 0

        return f1.mean().item()


class RobustJaccard:
    def __init__(self, dilation_size: int = 1, **params):
        self.dilation = batched_dilation
        self.dilation_size = dilation_size
        self.filter = GaussianFilter(**params)
        
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        batch_size, *other_axes = predicted.shape
        axes = list(range(1, len(other_axes) + 1))

        target = target.detach()
        predicted = predicted.detach()

        union = target + predicted
        union = torch.clip(union, 0 ,1)

        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        
        dilated_predicted = self.dilation(predicted.cpu(), self.dilation_size)
        dilated_predicted = dilated_predicted.to(device)

        intersec = dilated_predicted * target

        union = self.filter(union)
        intersec = self.filter(intersec)

        res = (torch.abs(torch.sum(intersec, dim = axes)) / torch.abs(torch.sum(union, dim = axes)))
        res[res != res] = 1 # Repalces NaN with 1

        return res.mean().item()

class Precision:
    def __init__(self, **params):
        self.p = torchmetrics.Precision(task = 'binary', average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:        
        target = target.detach()
        predicted = predicted.detach()
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        self.p = self.p.to(device)

        out = self.p(predicted, target)

        
        return out

class Recall:
    def __init__(self, **params):
        self.r = torchmetrics.Recall(task = 'binary', average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:        
        
        target = target.detach()
        predicted = predicted.detach()
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        self.r = self.r.to(device)
        out = self.r(predicted, target)
    
        
        return out


class Accuracy:
    def __init__(self):
        self.p = torchmetrics.Accuracy(task = 'binary', average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:        
        target = target.detach()
        predicted = predicted.detach()
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        self.p = self.p.to(device)
        out = self.p(predicted, target)
        
        return out


class Specificity:
    def __init__(self, **params):
        self.p = torchmetrics.Specificity(task = 'binary', average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:        
        target = target.detach()
        predicted = predicted.detach()
        
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        self.p = self.p.to(device)
        out = self.p(predicted, target)

        return out


class F05:
    def __init__(self, **params):
        pass
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor, precision = None, recall = None) -> float:
        
        if not (precision and recall):
            precision = Precision()(predicted, target)
            recall = Recall()(predicted, target)
        
        f05 = 1.5*precision*recall/(0.25*precision + recall + 1e-8)
        
        return f05


class F1:
    def __init__(self, **params):
        pass
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor, precision = None, recall = None) -> float:
        
        if not (precision and recall):
            precision = Precision()(predicted, target)
            recall = Recall()(predicted, target)
        
        f1 = 2*precision*recall/(precision + recall + 1e-8)
        
        return f1


class F2:
    def __init__(self, **params):
        pass
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor, precision = None, recall = None) -> float:
        
        if not (precision and recall):
            precision = Precision()(predicted, target)
            recall = Recall()(predicted, target)
        
        f2 = 5*precision*recall/(4*precision + recall + 1e-8)
        
        return f2


class AUROC:
    def __init__(self, **params):
        self.p = torchmetrics.AUROC(task = 'binary', thresholds = 64, average = 'samples')
    
    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> float:        
        
        target = target.detach()
        predicted = predicted.detach()
        
        rank = predicted.get_device()
        device = f'cuda:{rank}' if rank >= 0 else 'cpu'
        self.p = self.p.to(device)
        out = self.p(predicted, target)

        return out
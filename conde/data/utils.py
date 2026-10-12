import os
import matplotlib.pyplot as plt
import torch
from .metrics import *

def save_examples(x, recon, save_dir, save_name, max_images=8):
    """
    Saves side-by-side plots of input and reconstruction to self.save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    x = x.detach().cpu()
    recon = recon.detach().cpu()
    B = min(x.size(0), max_images)

    fig, axs = plt.subplots(B, 2, figsize=(9, 5*B))
    for i in range(B):
        axs[i][0].imshow(x[i][0].T, cmap="gray")
        axs[i][0].set_title("Original")
        axs[i][0].axis("off")

        axs[i][1].imshow(recon[i][0].T, cmap="gray")
        axs[i][1].set_title("Reconstruction")
        axs[i][1].axis("off")

    fname = os.path.join(save_dir, f"{save_name}_preds.png")     
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

def save_examples_n_pairs(
    x1: torch.Tensor,
    recon1: torch.Tensor,
    original_recon1: torch.Tensor,
    x2: torch.Tensor,
    recon2: torch.Tensor,
    original_recon2: torch.Tensor,
    save_dir: str,
    save_name: str,
    max_images: int = 8,
):
    """
    Save a PNG with an N x 3 grid:
        ┌────────────┬───────────────┬─────────────────────────┐
        │  x1[0]     │  recon1[0]    │  original_recon1[0]     │
        ├────────────┼───────────────┼─────────────────────────┤
        │  x2[0]     │  recon2[0]    │  original_recon2[0]     │
        ├────────────┼───────────────┼─────────────────────────┤
        │  x1[1]     │  recon1[1]    │  original_recon1[1]     │
        ├────────────┼───────────────┼─────────────────────────┤
        │  x2[1]     │  recon2[1]    │  original_recon2[1]     │
        └────────────┴───────────────┴─────────────────────────┘
    Each row alternates: x1/recon1/orig_recon1 then x2/recon2/orig_recon2.

    Parameters
    ----------
    x1, recon1, original_recon1, x2, recon2, original_recon2 : torch.Tensor
        Tensors with shape (N, C, H, W) or (N, H, W).
    save_dir : str
        Output folder.
    save_name : str
        Base filename (extension .png is added automatically).
    max_images : int
        Max number of pairs (rows) to plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    def _to_np(t: torch.Tensor, idx: int):
        t = t.detach().cpu()
        if t.ndim == 4:    # (N, C, H, W)
            t = t[idx, 0]  # first channel
        elif t.ndim == 3:  # (N, H, W)
            t = t[idx]
        return t.T         # transpose for nicer seismic display

    N = min(x1.size(0), x2.size(0), max_images)

    fig, axes = plt.subplots(
        2 * N, 3, figsize=(9, 2 * N * 1.5), squeeze=False
    )

    titles = ["Original", "Reconstruction", "Original\nReconstruction"]

    for i in range(N):
        row1_imgs = [_to_np(x1, i), _to_np(recon1, i), _to_np(original_recon1, i)]
        row2_imgs = [_to_np(x2, i), _to_np(recon2, i), _to_np(original_recon2, i)]

        for c, img in enumerate(row1_imgs):
            ax = axes[2 * i, c]
            ax.imshow(img, cmap="gray")
            if i == 0:
                ax.set_title(titles[c])
            ax.axis("off")

        for c, img in enumerate(row2_imgs):
            ax = axes[2 * i + 1, c]
            ax.imshow(img, cmap="gray")
            ax.axis("off")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{save_name}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def get_metrics(output, label):

    """
    Calculate a set of evaluation metrics for a given model output and label.
    
    Arguments:
    ---------
    - output (torch.Tensor): Predicted output tensor.
    - label (torch.Tensor): Ground truth tensor.
    
    Returns:
    --------
    - torch.Tensor : Tensor containing the calculated metric values in order:
        [Mean IoU, Gaussian F1, Robust Jaccard, Precision, Recall, Accuracy,
         Specificity, F0.5, F1, F2, AUROC].
    """


    miou = MeanIoU()
    gf1 = GaussianF1()
    rj = RobustJaccard()
    p = Precision()
    r = Recall()
    acc = Accuracy()
    sp = Specificity()
    f05 = F05()
    f1 = F1()
    f2 = F2()
    roc_auc = AUROC()

    # Calculate each metric
    metrics = [
        miou(output, label),
        gf1(output, label),
        rj(output, label),
        p(output, label),
        r(output, label),
        acc(output, label),
        sp(output, label),
        f05(output, label),
        f1(output, label),
        f2(output, label),
        roc_auc(output, label.int()),
    ]
    # Return the metrics as a NumPy array
    return torch.tensor(metrics)

def compute_pr_curve(predictions, labels, device):
    """
    Auxiliary function to compute the Precision-Recall curve used on the cut_off function, to find the cut-off that maximizes the F1 on Precision-Recall curve.

    Arguments:
    ----------
        - predictions (np.array):  Classes from prediction output.
        - labels (np.array): Correct labels for the output.
        - device (str): 'cuda' or 'cpu'.

    Returns:
    -------
        - (precision (torch.Tensor), recall (torch.Tensor), tholds (torch.Tensor)) : Three tensor representing the X-axis and Y-axis from Precision-Recall curve. Tholds represents the Threshold used for each point.
    
    """

    P = Precision(task = 'binary', average = 'samples')
    R = Recall(task = 'binary', average = 'samples')
    
    tholds = torch.linspace(0, 1, 32).to(device)
    precision = torch.zeros(size = (len(tholds),)).to(device)
    recall = torch.zeros(size = (len(tholds),)).to(device)

    predictions = predictions.view(labels.size(0), -1)
    labels = labels.view(labels.size(0), -1)
    
    for i, threshold in enumerate(tholds):
            
            pred = predictions.clone()
            pred[pred >= threshold] = 1
            pred[pred < threshold] = 0
            precision[i] = P(pred, labels)
            recall[i] = R(pred,labels)

    
    return precision, recall, tholds

def cut_off(precision: torch.Tensor, recall: torch.Tensor, tholds: torch.Tensor) -> float:
    """
    Auxiliary function to calculate the cut-off which maximizes the F1 for the Precision-Recall curve.
    
    Arguments
    ---------
        - precision (torch.Tensor): calculated precision for each threshold value.
        - recall (torch.Tensor): calculated recall for each threshold value.
        - tholds (torch.Tensor): a tensor with the threshold values.
    
    Returns
    -------
        - The threshold which maximizes the F1 value.
    """
    
    f1 = 2*precision*recall/(precision + recall)
    f1 = torch.nan_to_num(f1)
    
    idx = torch.argsort(f1[:-1], stable = True)[-1]
    idx = idx.item()

    return tholds[idx].item()
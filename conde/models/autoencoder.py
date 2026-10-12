import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler

from pytorch_msssim import ms_ssim
from ..data.utils import save_examples

from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from torch.optim.lr_scheduler import OneCycleLR

def get_onecycle_scheduler(
    optimizer,
    max_lr: float,
    total_steps: int = None,
    epochs: int = None,
    steps_per_epoch: int = None,
    warmup_steps: int = None,
    pct_start: float = None,
    anneal_strategy: str = 'cos',
    cycle_momentum: bool = True,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
):
    """
    Creates a OneCycleLR scheduler.

    You must supply either:
      - total_steps (int): total number of optimizer.step() calls, OR
      - epochs (int) AND steps_per_epoch (int).

    warmup_steps or pct_start determines the warmup fraction:
      - pct_start = warmup_steps / total_steps
      - or supply pct_start directly (float in (0,1)).
    """

    # Derive pct_start if warmup_steps was given
    if pct_start is None:
        if warmup_steps is None or total_steps is None:
            raise ValueError("Provide either pct_start or (warmup_steps and total_steps).")
        pct_start = warmup_steps / total_steps

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=-1)
    

def get_warmup_scheduler(optimizer, warmup_steps: int):
    """
    Linear warmup: LR grows linearly from 0 -> 1 (multiplier) over warmup_steps.
    """
    return LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))


def get_decay_scheduler(optimizer, step_size: int, gamma: float):
    """
    Step decay: multiplies LR by gamma every `step_size` steps.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_combined_scheduler(
    optimizer,
    warmup_steps: int,
    decay_step_size: int,
    decay_gamma: float
):
    """
    Sequential scheduler: warmup for `warmup_steps`, then step-decay.
    """
    warmup_sched = get_warmup_scheduler(optimizer, warmup_steps)
    decay_sched = get_decay_scheduler(optimizer, decay_step_size, decay_gamma)
    # milestones=[warmup_steps] tells SequentialLR to switch at that step
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, decay_sched],
        milestones=[warmup_steps]
    )


class Autoencoder(nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        loss_aggregator,
        lr = 1e-4,
        weight_decay = 0,
        warmup_steps: int = 1000,
        total_steps=100000,
        device: torch.device = None,
        clip_value=1,
        save_dir = 'res',
        mixed_precision=True
    ):
        super().__init__()
        self.model = backbone
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # instantiate optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer, self.scheduler= self.configure_optimizers()
        self.clip_value=clip_value
        
        self.loss_aggregator = loss_aggregator
        self.save_dir = save_dir

        self.global_step = 0
        

        self.emb_size = self.model.embed_dim

        self.mixed_precision=mixed_precision
        if mixed_precision:
            self.scaler = GradScaler(init_scale=2**16)
        

    def forward(self, x, return_emb=False):
        return self.model(x, noise_mu=0., return_emb=return_emb)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def get_pred(self, x, return_emb=False):
        return self(x, return_emb=return_emb)


    def calculate_loss(self, recon, x, z):
        input_lookup = {'recon': recon, 'x': x, 'z':z}
        loss, loss_dict = self.loss_aggregator(input_lookup)
        return loss, loss_dict


    def calculate_metrics(self, recon, target):
        """
        Computes evaluation metrics like MS-SSIM.
        Assumes input is in [0, 1] or normalized consistently.
        """
        '''with torch.no_grad():
            mssim = ms_ssim(recon, target, data_range=1.0, size_average=True).item()'''
        return {"mssim": 1}


    def training_step(self, batch, batch_idx):
        x = batch
        x = x.to(self.device)
        if self.mixed_precision:
            with autocast():
                recon, z = self.get_pred(x, return_emb=True)
                loss, loss_dict = self.calculate_loss(recon, x,z)
        else:
            recon, z = self.get_pred(x, return_emb=True)
            loss, loss_dict = self.calculate_loss(recon, x,z)

        self.optimizer_step(loss)

        return loss_dict

    def validation_step(self, batch, batch_idx, max_images=8, epoch = 0):
        """
        Evaluation step for validation or testing.
        Logs reconstruction loss and evaluation metrics, and saves sample reconstructions.
        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(self.device)
    
        with torch.no_grad():
            recon, z = self.get_pred(x, return_emb=True)
            loss, loss_dict = self.calculate_loss(recon, x,z)
            metrics = self.calculate_metrics(recon, x)
    
        # Save reconstructions
        if max_images > 0:
            save_examples(x, recon, save_dir = self.save_dir, save_name=f'recon_{epoch}_{batch_idx}', max_images=max_images)
    
        return {**loss_dict, **metrics}

    

    

    def configure_optimizers(self):
        enc_params, dec_params = [], []
        for name, p in self.model.named_parameters():
            if "decoder" in name:
                dec_params.append(p)
            else:
                enc_params.append(p)
        
        optimizer = torch.optim.AdamW([
            {"params": enc_params, "weight_decay": self.weight_decay},  # keep a little on encoder if you like
            {"params": dec_params, "weight_decay": 0.0},   # no decay on decoder
        ], lr=self.lr,)

        scheduler = get_onecycle_scheduler(
            optimizer,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            max_lr = self.lr
        )
        
        return optimizer, scheduler
    
    def optimizer_step(self, total_loss):

        self.optimizer.zero_grad()
        if self.mixed_precision:
            self.scaler.scale(total_loss).backward()
            

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
        self.scheduler.step()
    
    
        self.global_step += 1

    

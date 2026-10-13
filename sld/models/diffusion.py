"""Conditional latent diffusion with partial noising (DiffuSeq-style), 2D.

The state is z = z_x (+) z_y, concatenated along channels: z_x is the scaled
seismic latent (the condition) and z_y the scaled RGT latent (the target).
Only z_y is noised. The UNet sees cat(z_x, z_y_t) and predicts the noise (or v)
of the z_y channels only. Sampling starts z_y from pure noise and re-concatenates
the clean z_x at every reverse step, so the condition is an exact anchor.

Training uses a DDPM schedule; sampling uses DDIM on the same betas
(``eta=0`` deterministic, ``eta=1`` ~ ancestral). Optional condition dropout
(``cond_drop_prob``, z_x -> 0) enables classifier-free guidance at sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from .regressor import make_unet2d


class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, cond_channels=8, latent_channels=4, block_out_channels=(128, 256, 512), layers_per_block=2,
                 attention_levels=(0, 1, 2), norm_num_groups=32, attention_head_dim=64,
                 num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                 prediction_type="epsilon", cond_drop_prob=0.0):
        super().__init__()
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"prediction_type must be epsilon or v_prediction, got {prediction_type!r}")
        self.model_cfg = dict(cond_channels=cond_channels, latent_channels=latent_channels,
                              block_out_channels=list(block_out_channels), layers_per_block=layers_per_block,
                              attention_levels=sorted(set(attention_levels or ())), norm_num_groups=norm_num_groups,
                              attention_head_dim=attention_head_dim, num_train_timesteps=num_train_timesteps,
                              beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end,
                              prediction_type=prediction_type, cond_drop_prob=cond_drop_prob)
        self.cond_channels, self.latent_channels = cond_channels, latent_channels
        self.prediction_type, self.cond_drop_prob = prediction_type, cond_drop_prob
        self.unet = make_unet2d(cond_channels + latent_channels, latent_channels, block_out_channels, layers_per_block,
                                attention_levels, norm_num_groups, attention_head_dim)
        # clip_sample=False: latents are not bounded to [-1, 1].
        self._sched_kwargs = dict(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
                                  beta_start=beta_start, beta_end=beta_end, prediction_type=prediction_type,
                                  clip_sample=False)
        self.scheduler = DDPMScheduler(**self._sched_kwargs)

    @property
    def num_train_timesteps(self):
        return self.model_cfg["num_train_timesteps"]

    def forward(self, z_y_t, t, z_x):
        """Noise (or v) prediction for the z_y channels."""
        return self.unet(torch.cat([z_x, z_y_t], dim=1), t).sample

    def training_loss(self, z_x, z_y, generator=None):
        """MSE on the z_y channels; z_x is never noised."""
        b, dev = z_y.shape[0], z_y.device
        noise = torch.randn(z_y.shape, generator=generator, device=dev, dtype=z_y.dtype)
        t = torch.randint(0, self.num_train_timesteps, (b,), generator=generator, device=dev)
        z_y_t = self.scheduler.add_noise(z_y, noise, t)
        if self.training and self.cond_drop_prob > 0:
            drop = torch.rand(b, generator=generator, device=dev) < self.cond_drop_prob
            z_x = torch.where(drop.view(-1, 1, 1, 1), torch.zeros_like(z_x), z_x)
        target = noise if self.prediction_type == "epsilon" else self.scheduler.get_velocity(z_y, noise, t)
        return F.mse_loss(self(z_y_t, t, z_x).float(), target.float())

    @torch.no_grad()
    def sample(self, z_x, num_inference_steps=50, eta=0.0, guidance_scale=1.0, generator=None):
        """One z_y sample per row of z_x (repeat z_x for several samples). Returns scaled z_y, float32."""
        sched = DDIMScheduler(**self._sched_kwargs, set_alpha_to_one=True, timestep_spacing="trailing")
        sched.set_timesteps(num_inference_steps, device=z_x.device)
        b, _, h, w = z_x.shape
        z_x = z_x.float()
        z = torch.randn((b, self.latent_channels, h, w), generator=generator, device=z_x.device)
        guided = guidance_scale != 1.0
        cond = torch.cat([z_x, torch.zeros_like(z_x)]) if guided else z_x
        for t in sched.timesteps:
            out = self(torch.cat([z, z]) if guided else z, t, cond).float()  # clean z_x re-anchored every step
            if guided:
                out_c, out_u = out.chunk(2)
                out = out_u + guidance_scale * (out_c - out_u)
            z = sched.step(out, t, z, eta=eta, generator=generator).prev_sample
        return z

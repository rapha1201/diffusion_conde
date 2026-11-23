"""Conditional latent diffusion with partial noising (DiffuSeq-style), 2D.

The state is z = z_x (+) z_y, concatenated along channels: z_x is the scaled
seismic latent (the condition) and z_y the scaled RGT latent (the target).
Only z_y is noised. The UNet sees cat(z_x, z_y_t) and predicts the noise (or v)
of the z_y channels only. Sampling starts z_y from pure noise and re-concatenates
the clean z_x at every reverse step, so the condition is an exact anchor.

Training uses a DDPM schedule; sampling uses DDIM on the same betas
(``eta=0`` deterministic, ``eta=1`` ~ ancestral). Optional condition dropout
(``cond_drop_prob``, z_x -> 0) enables classifier-free guidance at sampling.

Optional condition encoder (``cond_embed_dim > 0``): a small CNN maps z_x to
``cond_embed_dim`` feature channels h and the UNet sees cat(z_x, h, z_y_t). It is meant to
be the seismic encoder pretrained with CLIP (sld/models/clip.py, train_clip.py) and loaded
by train_ldm.py, usually frozen, like the CLIP text encoder of Stable Diffusion.

Optional loss guidance at sampling (``sample(guide_fn=...)``): e.g. the seismic-RGT dip
consistency of sld/guidance.py, applied to the predicted clean latent. The weights are untouched.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from .latent_encoder import make_latent_encoder
from .regressor import make_unet2d


class ConditionalLatentDiffusion(nn.Module):
    def __init__(self, cond_channels=8, latent_channels=4, block_out_channels=(128, 256, 512), layers_per_block=2,
                 attention_levels=(0, 1, 2), norm_num_groups=32, attention_head_dim=64,
                 num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                 prediction_type="epsilon", cond_drop_prob=0.0,
                 cond_embed_dim=0, cond_encoder_blocks=4, cond_norm_groups=None):
        super().__init__()
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(f"prediction_type must be epsilon or v_prediction, got {prediction_type!r}")
        cond_norm_groups = cond_norm_groups or norm_num_groups
        self.model_cfg = dict(cond_channels=cond_channels, latent_channels=latent_channels,
                              block_out_channels=list(block_out_channels), layers_per_block=layers_per_block,
                              attention_levels=sorted(set(attention_levels or ())), norm_num_groups=norm_num_groups,
                              attention_head_dim=attention_head_dim, num_train_timesteps=num_train_timesteps,
                              beta_schedule=beta_schedule, beta_start=beta_start, beta_end=beta_end,
                              prediction_type=prediction_type, cond_drop_prob=cond_drop_prob,
                              cond_embed_dim=cond_embed_dim, cond_encoder_blocks=cond_encoder_blocks,
                              cond_norm_groups=cond_norm_groups)
        self.cond_channels, self.latent_channels = cond_channels, latent_channels
        self.prediction_type, self.cond_drop_prob = prediction_type, cond_drop_prob
        self.cond_encoder = (make_latent_encoder(cond_channels, cond_embed_dim, cond_encoder_blocks, cond_norm_groups)
                             if cond_embed_dim > 0 else None)
        self.unet = make_unet2d(cond_channels + max(cond_embed_dim, 0) + latent_channels, latent_channels,
                                block_out_channels, layers_per_block, attention_levels, norm_num_groups,
                                attention_head_dim)
        # clip_sample=False: latents are not bounded to [-1, 1].
        self._sched_kwargs = dict(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule,
                                  beta_start=beta_start, beta_end=beta_end, prediction_type=prediction_type,
                                  clip_sample=False)
        self.scheduler = DDPMScheduler(**self._sched_kwargs)

    @property
    def num_train_timesteps(self):
        return self.model_cfg["num_train_timesteps"]

    def condition(self, z_x):
        """What the UNet sees of the seismic: z_x, plus the condition-encoder features when enabled."""
        if self.cond_encoder is None:
            return z_x
        return torch.cat([z_x, self.cond_encoder(z_x)], dim=1)

    def forward(self, z_y_t, t, z_x):
        """Noise (or v) prediction for the z_y channels."""
        return self.unet(torch.cat([self.condition(z_x), z_y_t], dim=1), t).sample

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
    def sample(self, z_x, num_inference_steps=50, eta=0.0, guidance_scale=1.0, generator=None,
               guide_fn=None, guide_scale=0.0, guide_start=0.3, guide_stop=1.0, guide_inner=1):
        """One z_y sample per row of z_x (repeat z_x for several samples). Returns scaled z_y, float32.

        Optional loss guidance (``guide_fn``, ``guide_scale > 0``): at the steps whose progress
        i / num_inference_steps lies in [guide_start, guide_stop), the predicted clean latent
        z0 = (z_t - sqrt(1 - a_t) eps) / sqrt(a_t) takes ``guide_inner`` normalized gradient steps
        on guide_fn(z0) -> per-row loss [B] (e.g. the dip score of the decoded RGT), each moving its
        elements by ``guide_scale`` in RMS, and the noise prediction is recomputed from the new z0.
        Gradients go through guide_fn only (e.g. the frozen decoder), never through the UNet.
        """
        sched = DDIMScheduler(**self._sched_kwargs, set_alpha_to_one=True, timestep_spacing="trailing")
        sched.set_timesteps(num_inference_steps, device=z_x.device)
        b, _, h, w = z_x.shape
        z_x = z_x.float()
        z = torch.randn((b, self.latent_channels, h, w), generator=generator, device=z_x.device)
        guided = guidance_scale != 1.0
        cond = torch.cat([z_x, torch.zeros_like(z_x)]) if guided else z_x
        steer = guide_fn is not None and guide_scale > 0
        n_steps = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            out = self(torch.cat([z, z]) if guided else z, t, cond).float()  # clean z_x re-anchored every step
            if guided:
                out_c, out_u = out.chunk(2)
                out = out_u + guidance_scale * (out_c - out_u)
            if steer and guide_start <= i / n_steps < guide_stop:
                out = self._steer(sched, z, t, out, guide_fn, guide_scale, guide_inner)
            z = sched.step(out, t, z, eta=eta, generator=generator).prev_sample
        return z

    def _steer(self, sched, z, t, out, guide_fn, scale, inner):
        """Move the predicted z0 down the gradient of guide_fn, return the matching model output."""
        a = float(sched.alphas_cumprod[int(t)])
        sa, s1a = a ** 0.5, max(1.0 - a, 1e-12) ** 0.5
        if self.prediction_type == "epsilon":
            z0 = (z - s1a * out) / sa
        else:  # v-prediction: z0 = sqrt(a) z - sqrt(1-a) v
            z0 = sa * z - s1a * out
        with torch.enable_grad():
            for _ in range(max(1, int(inner))):
                x = z0.detach().requires_grad_(True)
                (g,) = torch.autograd.grad(guide_fn(x).sum(), x)
                g = g.float()
                rms = g.pow(2).flatten(1).mean(1).sqrt().clamp_min(1e-12).view(-1, 1, 1, 1)
                z0 = z0.detach() - scale * g / rms
        if self.prediction_type == "epsilon":
            return (z - sa * z0) / s1a
        return (sa * z - z0) / s1a

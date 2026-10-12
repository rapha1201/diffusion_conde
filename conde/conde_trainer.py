import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .losses import ExplainedVariance, MRRMetric, DecoderGap
from .utils import MetricAggregator
import math
import os
from scipy.io import wavfile

import torchjd
from torchjd.aggregation import UPGrad



class ConDeTrainer:
    """
    Generic trainer for ConDe with arbitrary semantic factors.

    Expected batch when require_labels=True:
        x1, x2, x3, labels, z1, z2, z3, pos_idx

    labels is the anchor RGT tensor. Samples without RGT must contain only -1.

    semantic_info: dict mapping:
        {
            "semantic_name": {"head_index": int}
        }
    """

    def __init__(
        self,
        conde_model,
        contrastive_loss_fn,
        residual_loss_fn,
        coverage_loss_fn = None,
        decoder_loss_fn = None,
        rgt_loss_fn=None,
        extract_embs = False,
        train_decoder = True,
        train_rgt_decoder = True,
        decoder=None,
        rgt_decoder=None,
        device="cuda",
        semantic_info=None,
        lambda_contrast=1.0,
        lambda_decomp=1.0,
        lambda_residual=1.0,
        lambda_coverage=1.0,
        lambda_decoder=1.0,
        lambda_rgt=1.0,
        decoder_warmup=0.25,
        lr = 1e-4,
        total_steps = 10000,
        grad_clip_norm=5,
        use_upgrad=False,
        decomp_mode = 'inverted', # inverted, inverted_detach, regular or regular_detach
        coverage_mode = 'inverted',  # inverted, z_sem, regular
        require_labels = False,

        save_dir=None
    ):
        self.model = conde_model
        self.decoder = decoder
        self.rgt_decoder = rgt_decoder
        self.extract_embs = extract_embs
        self.device = device

        self.semantic_info = semantic_info
        self.contrastive_loss_fn = contrastive_loss_fn
        self.residual_loss_fn = residual_loss_fn
        self.coverage_loss_fn = coverage_loss_fn if coverage_loss_fn is not None else F.mse_loss
        self.decoder_loss_fn=decoder_loss_fn
        self.rgt_loss_fn=rgt_loss_fn


        self.lambda_contrast = lambda_contrast
        self.lambda_decomp = lambda_decomp
        self.lambda_residual = lambda_residual
        self.lambda_coverage=lambda_coverage
        self.lambda_decoder=lambda_decoder
        self.lambda_rgt=lambda_rgt
        self.train_decoder = train_decoder and self.decoder_loss_fn is not None
        self.train_rgt_decoder = train_rgt_decoder and self.rgt_decoder is not None and self.rgt_loss_fn is not None

        for p in self.model.parameters():
            p.requires_grad = True

        # Decoder: train only if recon loss is used
        if self.decoder is not None:
            for p in self.decoder.parameters():
                p.requires_grad = self.train_decoder

        if self.rgt_decoder is not None:
            for p in self.rgt_decoder.parameters():
                p.requires_grad = self.train_rgt_decoder

        params = []

        params += list(self.model.parameters())

        if self.train_decoder:
            params += list(self.decoder.parameters())

        if self.train_rgt_decoder:
            params += list(self.rgt_decoder.parameters())

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                self.optimizer, 
                                max_lr=lr,                     # Peak LR
                                total_steps = total_steps,
                                pct_start=0.06
                            ) 
        self.total_steps=total_steps
        self.grad_clip_norm=grad_clip_norm

        self.loss_agg = MetricAggregator()

        self.val_loss_agg = MetricAggregator()
        self.metric_agg = MetricAggregator()


        self.ev_metric  = ExplainedVariance()
        self.mrr_metric = MRRMetric()
        self.gap_metric = DecoderGap(similarity=False)

        possible_decomp = ['inverted', 'inverted_detach', 'regular', 'regular_detach']
        assert decomp_mode in possible_decomp , f"Invalid decomp_mode, please choose between one of {possible_decomp} "

        possible_coverage =  ['inverted', 'inverted_detach', 'regular', 'z_sem', 'inverted_z_sem']
        assert coverage_mode in possible_coverage , f"Invalid decomp_mode, please choose between one of {possible_coverage} "

        self.decomp_mode = decomp_mode
        self.coverage_mode = coverage_mode

        self.require_labels = require_labels


        self.save_dir=save_dir

        self.use_upgrad=False
        if use_upgrad:
            
            self.jd_agg = UPGrad()
            self.use_upgrad=True

        self.decoder_warmup=decoder_warmup*total_steps + 1

        self.steps=0

    def add_noise(self, r, gate, start_std=0.0, end_std=0.5, end_pct=0.2):
        """
        Adds Mask-Scaled Gaussian noise to R with a linearly WARM-UP schedule.
        
        gate      : The gating activations, sigma(g(Z)). Shape must match r.
        start_std : Initial noise std (usually 0 to let the AE learn basic recon first)
        end_std   : Maximum noise std (the final bottleneck pressure)
        """
        # 1. Warm-up schedule (increase pressure over time, then hold constant)
        progress = min(self.steps / (self.total_steps * end_pct), 1.0)
        current_std = start_std * (1 - progress) + end_std * progress
        
        # 2. Generate the raw base noise
        raw_noise = torch.randn_like(r) * current_std
        
        # 3. THE TRAP: Scale the noise by the gate (or by R)
        # If gate -> 0, noise -> 0. The escape hatch is open.
        scaled_noise = raw_noise * gate 
        
        return r + scaled_noise

    def get_decoder_lambda(self):
        alpha = min(1, self.steps/self.decoder_warmup)

        return alpha*self.lambda_decoder

        

    def build_pos_neg(self, s1, s2, s3, pos_idx, head_idx):
        """
        s1, s2, s3 = embeddings from semantic head k on x1, x2, x3
        pos_idx[:, k] = Boolean that selects which is positive for x1
        """
        mask = pos_idx[:, head_idx].bool().unsqueeze(1).to(self.device)
        pos = torch.where(mask, s3, s2)
        neg = torch.where(mask, s2, s3)
        return pos, neg
    
    def on_epoch_end(self, do_print=True):

        if do_print:
            self.loss_agg.print()
        self.loss_agg.reset()

    def extract_preds(self, batch):

        labels = None
        if self.require_labels:
            x1, x2, x3, labels, z1, z2, z3, pos_idx = batch
        else:
            x1, x2, x3, z1, z2, z3, pos_idx = batch

        x1, x2, x3 = [t.to(self.device) for t in (x1, x2, x3)]
        z1, z2, z3 = [t.to(self.device) for t in (z1, z2, z3)]
        
        pos_idx = pos_idx.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        if self.extract_embs:

            x_cat = torch.cat([x1, x2, x3], dim=0)
            out_cat = self.model(x_cat)                 
            B = x1.shape[0]

            out1, out2, out3 = {}, {}, {}
            for k, v in out_cat.items():
                if isinstance(v, torch.Tensor):
                    # simple case: just split along batch dim
                    out1[k] = v[:B]
                    out2[k] = v[B:2*B]
                    out3[k] = v[2*B:]
                elif isinstance(v, (list, tuple)):
                    # list/tuple of tensors, e.g. S_list
                    out1[k] = [t[:B]       for t in v]
                    out2[k] = [t[B:2*B]    for t in v]
                    out3[k] = [t[2*B:]     for t in v]
        else:
            
            out1  = self.model(z1)
            out2  = self.model(z2)
            out3 = self.model(z3)



        S_sum = torch.stack(out1["S_list"], dim=1).sum(dim=1) 

        S2 = torch.stack(out2["S_list"], dim=1)
        S3 = torch.stack(out3["S_list"], dim=1)
        
        mask = pos_idx.bool().unsqueeze(-1)
        pos_S_sum = torch.where(mask, S3, S2).sum(dim=1)

        preds = {
            "x1": x1, "x2": x2, "x3": x3,
            "z1_in": z1, "z2_in": z2, "z3_in": z3,
            "pos_idx": pos_idx,
            "labels": labels,
            "out1": out1, "out2": out2, "out3": out3,
            "S_sum":S_sum, "pos_S_sum":pos_S_sum,
            

            # Parsed components (anchor + other views)
            "R1": out1["R"],  "Zs1": out1["Z_sem"], "S1": out1["S_list"], "z1_hat": out1["Z"], "gate1": out1["gate"],
            "R2": out2["R"],  "Zs2": out2["Z_sem"], "S2": out2["S_list"], "z2_hat": out2["Z"],
            "R3": out3["R"],  "Zs3": out3["Z_sem"], "S3": out3["S_list"], "z3_hat": out3["Z"],
        }
        return preds



    def attenuate_R(self, r, start_keep=1.0, end_keep=0.65, end_pct=0.2):

        # warmup schedule
        progress = min(self.steps / (self.total_steps * end_pct), 1.0)
        current_keep = start_keep * (1 - progress) + end_keep * progress

        alpha = torch.empty_like(r).uniform_(current_keep, 1.0)

        return r * alpha
    
    def get_loss(self, *, x1, R1, Zs1, Zs2, S1, S2, S3, z1_hat, pos_idx, S_sum, pos_S_sum, gate1, labels=None, train=True):

        def safe(t):
            return torch.where(torch.isfinite(t), t, t.new_zeros(()))
        
        total_contrast = 0.0
        losses_by_semantic = {}
        pos_semantics = []

        loss_dict = {}

        if self.use_upgrad and train==True:
            loss_list = []
            
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
            s1 = S1[k]
            s2 = S2[k]
            s3 = S3[k]

            pos, neg = self.build_pos_neg(s1, s2, s3, pos_idx, k)

            loss_k = self.contrastive_loss_fn(s1, pos, neg)
            loss_k = safe(loss_k)

            losses_by_semantic[sem_name] = loss_k.detach().item()
            if self.use_upgrad and train==True:
                loss_list.append(loss_k)
            total_contrast = total_contrast + loss_k
            pos_semantics.append(pos)

        loss_dict['contrast_total'] = total_contrast
        loss_dict['contrast_by_semantic'] = losses_by_semantic
        

        # -------------------------
        # Decomposition loss
        # -------------------------
        if self.decomp_mode == "inverted":
            decomp_loss = F.mse_loss(Zs1, pos_S_sum)
        elif self.decomp_mode == "inverted_detach":
            decomp_loss = F.mse_loss(Zs1.detach(), pos_S_sum)
        elif self.decomp_mode == "regular_detach":
            decomp_loss = F.mse_loss(Zs1.detach(), S_sum)
        else:
            decomp_loss = F.mse_loss(Zs1, S_sum)
        
        decomp_loss = safe(decomp_loss)

        loss_dict['decomp'] = decomp_loss
        if self.use_upgrad and train==True:
            loss_list.append(decomp_loss)


        # -------------------------
        # Coverage loss
        # -------------------------

        if self.coverage_mode == "inverted":
            coverage_loss = self.coverage_loss_fn(pos_S_sum, z1_hat)
        elif self.coverage_mode == "inverted_detach":
            coverage_loss = self.coverage_loss_fn(pos_S_sum, z1_hat.detach())
        elif self.coverage_mode == "z_sem":
            coverage_loss = self.coverage_loss_fn(Zs1, z1_hat)
        elif self.coverage_mode == "inverted_z_sem":
            # Change later to use positive idx, not needed for now
            coverage_loss = self.coverage_loss_fn(Zs2, z1_hat)
        else:
            coverage_loss = self.coverage_loss_fn(S_sum, z1_hat)

        coverage_loss=safe(coverage_loss)

        loss_dict['coverage'] = coverage_loss
        if self.use_upgrad and train==True:
            loss_list.append(coverage_loss)


        residual_loss = z1_hat.new_tensor(0.0)
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
            residual_loss = residual_loss + safe(self.residual_loss_fn(R1, S1[k].detach()))
        
        loss_dict['residual'] = residual_loss
        if self.use_upgrad and train==True:
            loss_list.append(residual_loss)

        

        decoder_loss = z1_hat.new_tensor(0.0)
        if self.decoder_loss_fn is not None:
            
            X_rec = self.decoder(Zs1 + R1)
            X_rec_inv = self.decoder(pos_S_sum + R1.detach())
            decoder_loss = safe(self.decoder_loss_fn(X_rec, x1)) + safe(self.decoder_loss_fn(X_rec_inv, x1))
            #z_noisy = self.add_latent_noise(z1_hat, noise_std=0.02)
            #X_rec = self.decoder(z1_hat)
            #pos_rec = self.decoder(pos_S_sum + R1)
            #decoder_loss = safe(self.decoder_loss_fn(pos_rec, x1))
            if self.use_upgrad and train==True:
                loss_list.append(decoder_loss)

        loss_dict['recon'] = decoder_loss

        # -------------------------
        # RGT loss
        # -------------------------
        rgt_loss = z1_hat.new_tensor(0.0)
        rgt_pred = None
        rgt_targets = None

        if self.rgt_decoder is not None and self.rgt_loss_fn is not None and labels is not None:
            valid_rgt = labels.ne(-1).reshape(labels.shape[0], -1).any(dim=1)

            if valid_rgt.any():
                rgt_targets = labels[valid_rgt]
                rgt_pred = self.rgt_decoder(pos_S_sum[valid_rgt])
                rgt_loss = safe(self.rgt_loss_fn(rgt_pred, rgt_targets))

                if self.use_upgrad and train == True:
                    loss_list.append(rgt_loss)

        loss_dict['rgt'] = rgt_loss

        z_norm = safe(z1_hat.pow(2).mean())
        loss_dict['z_norm'] = z_norm

        r_norm = safe(gate1.mean())
        loss_dict['r_norm'] = r_norm

        # -------------------------
        # Total loss
        # -------------------------
        total_loss = (
            self.lambda_contrast * total_contrast
            + self.lambda_decomp * decomp_loss
            + self.lambda_residual * residual_loss
            + self.lambda_coverage * coverage_loss
            + (self.get_decoder_lambda() * decoder_loss if self.decoder_loss_fn is not None else 0.0)
            + self.lambda_rgt * rgt_loss
            + z_norm * 0.05
            + r_norm * 0.05
        )

        loss_dict['total'] = total_loss

        if self.use_upgrad and train==True:
            return loss_dict, loss_list
        return loss_dict


    def train_step(self, batch):

        self.steps+=1
        
        self.model.train()
        if self.train_decoder:
            self.decoder.train()
        if self.train_rgt_decoder:
            self.rgt_decoder.train()


        preds = self.extract_preds(batch)        

        self.optimizer.zero_grad()

        if self.use_upgrad:
            loss_dict, jd_losses = self.get_loss(
                x1=preds["x1"],
                R1=preds["R1"],
                Zs1=preds["Zs1"],
                Zs2=preds["Zs2"],
                S1=preds["S1"],
                S2=preds["S2"],
                S3=preds["S3"],
                z1_hat=preds["z1_hat"],
                S_sum = preds["S_sum"],
                pos_S_sum = preds["pos_S_sum"],
                gate1 = preds['gate1'],
                pos_idx=preds["pos_idx"],
                labels=preds["labels"],
                train=True,
            )
                
            torchjd.backward(jd_losses, self.jd_agg)

        else:
            loss_dict = self.get_loss(
                x1=preds["x1"],
                R1=preds["R1"],
                Zs1=preds["Zs1"],
                Zs2=preds["Zs2"],
                S1=preds["S1"],
                S2=preds["S2"],
                S3=preds["S3"],
                z1_hat=preds["z1_hat"],
                S_sum = preds["S_sum"],
                pos_S_sum = preds["pos_S_sum"],
                gate1 = preds['gate1'],
                pos_idx=preds["pos_idx"],
                labels=preds["labels"],
                train=True
            )
            loss_dict["total"].backward()

        if self.grad_clip_norm is not None:
            params = list(self.model.parameters())
            if self.train_decoder:
                params += list(self.decoder.parameters())
            if self.train_rgt_decoder:
                params += list(self.rgt_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()

        stats = {
            "total": loss_dict["total"].detach().item(),
            "decomp": loss_dict["decomp"].detach().item(),
            "residual": loss_dict["residual"].detach().item(),
            "coverage_loss": loss_dict["coverage"].detach().item(),
            "z_norm": loss_dict["z_norm"].detach().item(),
            "contrast_by_semantic": loss_dict["contrast_by_semantic"],
            "rgt": loss_dict["rgt"].detach().item(),
        }
        if self.decoder_loss_fn is not None:
            stats["recon"] = loss_dict["recon"].detach().item()

        self.loss_agg.update(stats)
        return stats


    @torch.no_grad()
    def validation_step(self, batch, n_show=5, save_name='res'):
        self.model.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.rgt_decoder is not None:
            self.rgt_decoder.eval()


        preds = self.extract_preds(batch)

        loss_dict = self.get_loss(
            x1=preds["x1"],
            R1=preds["R1"],
            Zs1=preds["Zs1"],
            Zs2=preds["Zs2"],
            S1=preds["S1"],
            S2=preds["S2"],
            S3=preds["S3"],
            z1_hat=preds["z1_hat"],
            S_sum = preds["S_sum"],
            pos_S_sum = preds["pos_S_sum"],
            gate1 = preds['gate1'],
            pos_idx=preds["pos_idx"],
            labels=preds["labels"],
            train=False,
        )

        self.val_loss_agg.update({
            "total": loss_dict["total"].item(),
            "decomp": loss_dict["decomp"].item(),
            "residual": loss_dict["residual"].item(),
            "contrast_by_semantic": loss_dict["contrast_by_semantic"],
            "coverage_loss": loss_dict["coverage"].item(),
            "rgt": loss_dict["rgt"].item(),
            **({"recon": loss_dict["recon"].item()} if self.decoder_loss_fn is not None else {}),
        }, n=preds["z1_hat"].shape[0])

        # --- Online metrics (keep your existing logic, but reuse values from preds/loss_dict)
        S1_sum = sum(preds["S1"])
        EV_sem, EV_full = self.ev_metric(preds["z1_hat"], S1_sum, preds["R1"])

        # These already match your prior validation code
        X_ae = self.decoder(preds["z1_hat"])
        X_sr = self.decoder(preds["pos_S_sum"] + preds["R1"])
        dec_gap = self.gap_metric(X_ae, X_sr)

        mrr_list = []
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
            sA = preds["S1"][k]
            sP = preds["S3"][k]
            mrr_list.append(self.mrr_metric(sA, sP))
        mrr_mean = sum(mrr_list) / len(mrr_list)

        self.metric_agg.update({
            "mrr": mrr_mean,
            "EV_sem": EV_sem,
            "EV_full": EV_full,
            "decoder_gap": dec_gap
        }, n=preds["z1_hat"].shape[0])

        if n_show > 0:
            batch_for_vis = (preds["x1"], preds["x2"], preds["x3"], preds["z1_hat"], preds["z2_hat"], preds["z3_hat"], preds["pos_idx"])
            self._visualize_val(batch_for_vis, preds["out1"], preds["out2"], preds["out3"], labels=preds["labels"], pos_S_sum=preds["pos_S_sum"], n_show=n_show, save_name=save_name)
            self.did_visualize = True

    @torch.no_grad()
    def _visualize_val(self, batch, out1, out2, out3, labels, pos_S_sum, n_show=5, save_name="res"):
        x1, x2, x3, z1, z2, z3, pos_idx = batch

        R1, Zs1, S1 = out1["R"], out1["Z_sem"], out1["S_list"]
        R2, Zs2, S2 = out2["R"], out2["Z_sem"], out2["S_list"]
        R3, Zs3, S3 = out3["R"], out3["Z_sem"], out3["S_list"]

        n_show = min(n_show, x1.shape[0])

        rec_dict = {
            "orig_x1": x1,
            "orig_x2": x2,
            "orig_x3": x3,
            "orig_x1_rec": self.decoder(z1),
            "orig_x2_rec": self.decoder(z2),
            "orig_x3_rec": self.decoder(z3),
        }

        rec_dict["conde_rec"] = self.decoder(sum(S1) + R1)

        # Swapped reconstructions for each semantic head.
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]

            s2 = S2[k]
            s3 = S3[k]

            mask = pos_idx[:, k].bool().unsqueeze(1)

            pos = torch.where(mask, s3, s2)
            neg = torch.where(mask, s2, s3)

            other_sum = sum(S1[j] for j in range(len(S1)) if j != k)

            swapped_pos = self.decoder(pos + other_sum + R1)
            swapped_neg = self.decoder(neg + other_sum + R1)

            rec_dict[f"{sem_name}_pos"] = swapped_pos
            rec_dict[f"{sem_name}_neg"] = swapped_neg

        rec_dict["R_solo"] = self.decoder(R1)

        # Start with black RGT images. Unlabeled samples remain black and are
        # never passed through rgt_decoder.
        if labels is None:
            rgt_labels = torch.zeros_like(x1)
            rgt_predictions = torch.zeros_like(x1)
        else:
            rgt_labels = torch.zeros_like(labels)
            rgt_predictions = torch.zeros_like(labels)

            rgt_predictions = self.rgt_decoder(pos_S_sum)

            valid_rgt = labels.ne(-1).reshape(labels.shape[0], -1).any(dim=1)

            if valid_rgt.any():
                rgt_labels[valid_rgt] = labels[valid_rgt]

        rec_dict["rgt_pred"] = rgt_predictions
        rec_dict["rgt_label"] = rgt_labels

        cols = len(rec_dict)
        fig, axes = plt.subplots(n_show, cols, figsize=(cols * 2, n_show * 2), squeeze=False)

        for row in range(n_show):
            for col, (name, data) in enumerate(rec_dict.items()):
                ax = axes[row, col]
                img = data[row].detach().squeeze().cpu().numpy()

                ax.imshow(img.T, cmap="gray")
                ax.axis("off")

                if row == 0:
                    ax.set_title(name, fontsize=10)

        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()
        

    def on_validation_epoch_end(self):

        print("\n[Validation Losses]")
        self.val_loss_agg.print()
        print("\n[Validation Metrics]")
        self.metric_agg.print()

        self.val_loss_agg.reset()
        self.metric_agg.reset()
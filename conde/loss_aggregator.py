import torch
import torch.nn as nn
from .losses import get_loss

class LossAggregator(nn.Module):
    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.losses = []
        for key, loss_cfg in config.items():
            loss_name = loss_cfg["name"]
            loss_class = get_loss(loss_name)
            if loss_class is None:
                raise ValueError(f"Unknown loss type: {loss_name!r}")
            args = loss_cfg.get("args", {})
            fn = loss_class(**args).to(self.device)
            alpha = loss_cfg.get("alpha", 1.0)
            pred = loss_cfg.get("pred", None)
            detach_pred = loss_cfg.get("detach_pred", False)
            assert pred is not None, "A loss function must be calculated on top of a prediction"
            target = loss_cfg.get('target', None)
            detach_target = loss_cfg.get("detach_target", False)
            triple = loss_cfg.get('triple', None)
            requires_step = loss_cfg.get('requires_step', False)
            self.losses.append({
                "key": key,
                "fn": fn,
                "alpha": alpha,
                "detach_target": detach_target,
                "detach_pred": detach_pred,
                "pred_key": pred,
                "target_key": target,
                "triple": triple,
                "requires_step": requires_step
            })

    def forward(self, input_lookup, step=None):
        total_loss = None
        loss_dict = {}

        
        for loss in self.losses:


            raw_pred = input_lookup[loss["pred_key"]]
            raw_target = input_lookup[loss["target_key"]] if loss["target_key"] else None
            if raw_pred is None or (loss["target_key"] and raw_target is None):
                loss_dict[loss['key']] = 0
                continue
        
            pred = raw_pred.detach() if loss["detach_pred"] else raw_pred
            target = raw_target.detach() if loss["detach_target"] else raw_target

            if loss['requires_step']:
                if loss["triple"] is not None:
                    triple = input_lookup[loss["triple"]]
                    loss_val = loss["fn"](pred, target, triple, step=step) * loss["alpha"]
                elif target is None:
                    loss_val = loss["fn"](pred, step=step) * loss["alpha"]
                else:
                    loss_val = loss["fn"](pred, target, step=step) * loss["alpha"]
            else:
                if loss["triple"] is not None:
                    triple = input_lookup[loss["triple"]]
                    loss_val = loss["fn"](pred, target, triple) * loss["alpha"]
                elif target is None:
                    loss_val = loss["fn"](pred) * loss["alpha"]
                else:
                    loss_val = loss["fn"](pred, target) * loss["alpha"]
                    
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"[WARN] NaN/Inf in loss '{loss['key']}' – replacing with 0")
                print(pred.max(), pred.min(), target.max(), target.min())
                loss_val = torch.zeros_like(loss_val)
                
            total_loss = loss_val if total_loss is None else total_loss + loss_val
            loss_dict[loss["key"]] = loss_val.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

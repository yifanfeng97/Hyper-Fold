"""Miscellaneous training utilities: distributed guard and EMA of weights."""
import copy

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    """Whether torch.distributed is both available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class ModelEMA:
    """Exponential moving average of model weights (params + float buffers).

    Eval and best-checkpoint selection use the EMA weights, which smooths out
    BN running-stat jitter and Hungarian-matching gradient noise.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # ramp the decay up over the first ~1000 steps so the EMA does not
        # lag the (fast-moving) early training phase
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_state = self.module.state_dict()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                ema_state[k].copy_(v)

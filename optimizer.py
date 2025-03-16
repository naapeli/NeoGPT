import torch

from math import cos, pi


class CosineAnnealingWithWarmup:
    def __init__(self, min_lr, max_lr, warmup_time, max_time):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_time = warmup_time
        self.max_time = max_time
    
    def get_lr(self, step):
        if step < self.warmup_time:
            return self.max_lr * step / self.warmup_time
        elif step > self.max_time:
            return self.min_lr
        decay_ratio = (step - self.warmup_time) / (self.max_time - self.warmup_time)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + cos(pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


def configure_optimizer(model_params, weight_decay, learning_rate):
    param_dict  = {pn: p for pn, p in model_params if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nondecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nondecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)  # use fused as it is faster but not the default option
    return optimizer


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    scheduler = CosineAnnealingWithWarmup(3e-5, 3e-4, 10, 50)

    x = np.arange(100) + 1
    y = []
    for step in x:
        y.append(scheduler.get_lr(step))

    plt.plot(x, y)
    plt.show()

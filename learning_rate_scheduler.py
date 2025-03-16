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

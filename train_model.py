from dataset import ShakespeareLoader
from GPT2 import GPT2, Config
from learning_rate_scheduler import CosineAnnealingWithWarmup

import torch

from time import perf_counter


dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

B, T = 4, 32
dataloader = ShakespeareLoader(B, T)

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

torch.set_float32_matmul_precision("high")

model = GPT2(Config())
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50
lr_scheduler = CosineAnnealingWithWarmup(min_lr, max_lr, warmup_steps, max_steps)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(1, 50 + 1):
    t0 = perf_counter()
    x, y = next(dataloader)
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=dev, dtype=torch.bfloat16):
    #     logits, loss = model(x, targets=y)
    logits, loss = model(x, targets=y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    # torch.cuda.synchronize()
    t1 = perf_counter()
    dt = (t1 - t0) * 1000
    print(f"Step {step} | loss: {loss.item():.6f} | norm of gradients: {norm:.2f} | lr: {lr:.2e} | tokens / sec: {B * T / (t1 - t0):.2f} | dt: {dt:.2f} ms")

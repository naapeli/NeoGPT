from dataset import ShakespeareLoader
from GPT2 import GPT2, Config
from optimizer import CosineAnnealingWithWarmup, configure_optimizer

import torch
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel

from time import perf_counter
import os

# ============ DEVICE ============
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="gloo")  # backend="nccl"  # gloo for amd gpu or cpu and nccl for nvidia gpu
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

# ============ DATALOADER ============
total_batch_size = 256  # 524288
B = 4  # 64
T = 32  # 2048
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f"Accumulation steps in training: {grad_accum_steps}")
dataloader = ShakespeareLoader(B, T, ddp_local_rank, ddp_world_size)

# ============ SEED AND PRECISION ============
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
torch.set_float32_matmul_precision("high")

# ============ MODEL ============
model = GPT2(Config())
model.to(device)
# model = torch.compile(model)
if ddp: model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

# ============ OPTIMIZER ============
max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50
lr_scheduler = CosineAnnealingWithWarmup(min_lr, max_lr, warmup_steps, max_steps)
optimizer = configure_optimizer(model.named_parameters(), weight_decay=0.1, learning_rate=6e-4)

# ============ TRAINING LOOP ============
for step in range(1, 50 + 1):
    t0 = perf_counter()
    total_loss = 0
    for sub_step in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # with torch.autocast(device_type=dev, dtype=torch.bfloat16):
        #     logits, loss = model(x, targets=y)
        logits, loss = model(x, targets=y)
        loss = loss / grad_accum_steps
        total_loss += loss.item()
        if ddp: model.require_backward_grad_sync = sub_step == grad_accum_steps - 1  # only sync the gradients on the last iteration of the accumulation loop
        loss.backward()

    if ddp: all_reduce(total_loss, op=ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    # torch.cuda.synchronize()
    t1 = perf_counter()
    dt = t1 - t0
    tokens_processed = grad_accum_steps * B * T * ddp_world_size
    if master_process: print(f"Step {step:5d} | loss: {total_loss:.6f} | norm of gradients: {norm:.2f} | lr: {lr:.3e} | tokens / sec: {tokens_processed / dt:.2f} | dt: {1000 * dt:.2f} ms")

if ddp: destroy_process_group()

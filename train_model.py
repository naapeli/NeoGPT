from dataset import ShakespeareLoader, FineWebLoader
from GPT2 import GPT2, Config
from optimizer import CosineAnnealingWithWarmup, configure_optimizer
from hellaswag import iterate_examples, render_example, get_most_likely_row

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel
import tiktoken

from time import perf_counter
import os

# ============ DEVICE ============
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # torchrun --standalone --nproc_per_node=1 train_model.py
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")  # backend="nccl"  # gloo for amd gpu or cpu and nccl for nvidia gpu
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # python train_model.py
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
device = torch.device(device)

# ============ DATALOADER ============
total_batch_size = 256  # 524288
B = 4  # 64
T = 32  # 2048
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: print(f"Accumulation steps in training: {grad_accum_steps}")
# trainloader = ShakespeareLoader(B, T, ddp_local_rank, ddp_world_size)
# validationloader = ShakespeareLoader(B, T, ddp_local_rank, ddp_world_size)
trainloader = FineWebLoader(B, T, ddp_local_rank, ddp_world_size, "train")
validationloader = FineWebLoader(B, T, ddp_local_rank, ddp_world_size, "val")

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
warmup_steps = 10  # 200
max_steps = 50  # 19073
lr_scheduler = CosineAnnealingWithWarmup(min_lr, max_lr, warmup_steps, max_steps)
optimizer = configure_optimizer(model.named_parameters(), weight_decay=0.1, learning_rate=6e-4)

# ============ LOGGING ============
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass
if master_process:
    writer = SummaryWriter(log_dir="runs/training")

def model_validation(step):
    validationloader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = validationloader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        all_reduce(val_loss_accum, op=ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        writer.add_scalar("Loss/val", val_loss_accum.item(), step)
            

def save_model():
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': model.module.state_dict() if ddp else model.state_dict(),
        'config': model.module.config if ddp else model.config,
        'step': step
    }
    torch.save(checkpoint, checkpoint_path)

def evaluate_hella_swag(step):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            #     logits, _ = model(tokens)
            logits, _ = model(tokens)
            pred_norm, _, _ = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        all_reduce(num_total, op=ReduceOp.SUM)
        all_reduce(num_correct_norm, op=ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")
        writer.add_scalar("HellaSwag accuracy", acc_norm, step)

def generate_text():
    num_return_sequences = 4
    max_length = 32
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        with torch.no_grad():
            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            #     logits, _ = model(xgen)
            logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")

# ============ TRAINING LOOP ============
for step in range(1, max_steps + 1):
    t0 = perf_counter()

    # validation
    if step % 250 or step == max_steps:
        model.eval()
        model_validation(step)
        evaluate_hella_swag(step)
        generate_text()
        model.train()
    if master_process and (step % 5000 == 0 or step == max_steps):
        save_model()

    total_loss = 0
    for sub_step in range(grad_accum_steps):
        x, y = trainloader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp: model.require_backward_grad_sync = sub_step == grad_accum_steps - 1  # only sync the gradients on the last iteration of the accumulation loop
        optimizer.zero_grad()
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #     logits, loss = model(x, targets=y)
        logits, loss = model(x, targets=y)
        loss = loss / grad_accum_steps
        total_loss += loss.detach()
        loss.backward()

    if ddp: all_reduce(total_loss, op=ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if device_type == "cuda": torch.cuda.synchronize()
    t1 = perf_counter()
    dt = t1 - t0
    tokens_processed = grad_accum_steps * B * T * ddp_world_size
    if master_process:
        print(f"Step {step:5d} | loss: {total_loss.item():.6f} | norm of gradients: {norm:.2f} | lr: {lr:.3e} | tokens / sec: {tokens_processed / dt:.2f} | dt: {1000 * dt:.2f} ms")
        with open(log_file, "a") as f:
            f.write(f"{step} train {total_loss.item():.6f}\n")
        writer.add_scalar("Loss/train", total_loss.item(), step)

if master_process: writer.close()
if ddp: destroy_process_group()

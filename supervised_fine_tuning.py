import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import os
import tiktoken

from GPT2 import GPT2
from dataset import InstructionDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

log_dir = "fine_tune_log"
checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
assert len(checkpoint_files) > 0, "no checkpoints found to resume training from"
checkpoint_files = sorted(checkpoint_files)
checkpoint_file = checkpoint_files[-1]
checkpoint_path = os.path.join(log_dir, checkpoint_file)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = GPT2(checkpoint["config"])
model.to(device)
model.load_state_dict(checkpoint["model"])

tokenizer = tiktoken.get_encoding("gpt2")
MAX_LENGTH = 1024  # use 150 for debugging and 1024 otherwise
BATCH_SIZE = 12
EPOCHS = 3
LR = 3e-5
LOG_INTERVAL = 10
SAVE_INTERVAL = 1000
SAVE_DIR = "fine_tune_log"
checkpoint_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("model_") and f.endswith(".pt")]
assert len(checkpoint_files) > 0, "no checkpoints found to resume training from"
checkpoint_files = sorted(checkpoint_files)
checkpoint_file = checkpoint_files[-1]
print(checkpoint_file)
checkpoint_path = os.path.join(SAVE_DIR, checkpoint_file)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

train_dataset = InstructionDataset(max_length=MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# for i in range(10):
#     print(train_dataset[i]["input_ids"], train_dataset[i]["input_ids"].shape)
#     print(train_dataset[i]["labels"], train_dataset[i]["labels"].shape)
#     print(train_dataset[i]["attention_mask"], train_dataset[i]["attention_mask"].shape)
# raise RuntimeError()

optimizer = AdamW(model.parameters(), lr=LR)

if "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])

start_epoch = checkpoint.get("epoch", 1)
start_step = checkpoint.get("step", 1)

model.train()

global_step = start_step
total_loss = 0
for epoch in range(start_epoch, EPOCHS + 1):
    for batch_idx, batch in enumerate(train_loader):
        global_step += 1
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        logits, loss = model(idx=input_ids, targets=labels, attn_mask=attention_mask)

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if global_step % LOG_INTERVAL == 0:
            log_message = f"Step {global_step} / {len(train_loader) * (EPOCHS + 1)}, Train loss: {total_loss}"
            print(log_message)
            with open(SAVE_DIR + "/log.txt", "a") as f:
                f.write(log_message + "\n")
            total_loss = 0

        if global_step % SAVE_INTERVAL == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": checkpoint["config"],
                "epoch": epoch,
                "step": global_step
            }, os.path.join(SAVE_DIR, f"model_{global_step}.pt"))

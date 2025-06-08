import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

import os

from GPT2 import GPT2
from dataset import PreferenceDataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

MAX_LENGTH = 100
BATCH_SIZE = 12
EPOCHS = 3
LR = 3e-5
BETA = 0.1
LOG_INTERVAL = 10
SAVE_INTERVAL = 1000
dataset = PreferenceDataset(MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def log_prob_sum(model, input_ids, attention_mask, response_mask):
    logits, _ = model(idx=input_ids, attn_mask=attention_mask)
    logits = logits[:, :-1]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    selected = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    return (selected * response_mask).sum(dim=1)

def criterion(policy_model, reference_model, batch):  # Direct preference optimization loss
    positive_prompt_ids = batch["positive_prompt_ids"].to(device)
    positive_attention_mask = batch["positive_attention_mask"].to(device)
    positive_response_mask = batch["positive_response_mask"].to(device)
    negative_prompt_ids = batch["negative_prompt_ids"].to(device)
    negative_attention_mask = batch["negative_attention_mask"].to(device)
    negative_response_mask = batch["negative_response_mask"].to(device)

    pos_logpi = log_prob_sum(policy_model, positive_prompt_ids, positive_attention_mask, positive_response_mask)
    neg_logpi = log_prob_sum(policy_model, negative_prompt_ids, negative_attention_mask, negative_response_mask)
    with torch.no_grad():
        pos_logref = log_prob_sum(reference_model, positive_prompt_ids, positive_attention_mask, positive_response_mask)
        neg_logref = log_prob_sum(reference_model, negative_prompt_ids, negative_attention_mask, negative_response_mask)
    pi_diff = pos_logpi - neg_logpi
    ref_diff = pos_logref - neg_logref
    logits = BETA * (pi_diff - ref_diff)
    loss = -torch.log(torch.sigmoid(logits)).mean()
    return loss

log_dir = "DPO_log"
checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
assert len(checkpoint_files) > 0, "no checkpoints found to resume training from"
checkpoint_files = sorted(checkpoint_files)
checkpoint_file = checkpoint_files[-1]
checkpoint_file2 = checkpoint_files[0]
print("policy model: ", checkpoint_file, "reference model: ", checkpoint_file2)
checkpoint_path = os.path.join(log_dir, checkpoint_file)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
checkpoint2 = torch.load(os.path.join(log_dir, checkpoint_file2), map_location=device, weights_only=False)
policy_model = GPT2(checkpoint["config"])
policy_model.to(device)
policy_model.load_state_dict(checkpoint["model"])
policy_model.train()

reference_model = GPT2(checkpoint2["config"])
reference_model.to(device)
reference_model.load_state_dict(checkpoint2["model"])
reference_model.eval()

optimizer = AdamW(policy_model.parameters(), lr=LR)

start_epoch = checkpoint.get("epoch", 1) + 1

for epoch in range(start_epoch, EPOCHS + 1):
    for i, batch in enumerate(dataloader):
        loss = criterion(policy_model, reference_model, batch)
        loss.backward()
        clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        log_message = f"Epoch {epoch} / {EPOCHS} - Step: {i + 1} / {len(dataloader)} - Loss: {loss.item()}"
        print(log_message)
        with open(log_dir + "/log.txt", "a") as f:
            f.write(log_message + "\n")
    
    # save the policy model at the end of every epoch
    print("Saving policy model")
    torch.save({
        "model": policy_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": checkpoint["config"],
        "epoch": epoch,
    }, os.path.join(log_dir, f"model_{epoch}.pt"))

import torch
from torch.utils.data import Dataset
import tiktoken
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os


def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def save_fineweb():
    local_dir = "edu_fineweb10B"
    remote_name = "sample-10BT"
    shard_size = int(1e8)

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    nprocs = max(1, os.cpu_count() - 1)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

class FineWebLoader:
    def __init__(self, B, T, process_rank, world_size, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size
        assert split in {"train", "val"}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.rng = np.random.default_rng(0)
        self.split = split
        self.reset()

    def reset(self):
        self.current_shard = 0
        if self.split == "train":
            self.rng.shuffle(self.shards)
        self.tokens = self.load_shard(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def load_shard(self, filename):
        shard = self.load_tokens(filename)
        if self.split == "train":
            enc = tiktoken.get_encoding("gpt2")
            eot = enc._special_tokens["<|endoftext|>"]
            eot_positions = (torch.where(shard == eot)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents)
        return shard
    
    def load_tokens(self, filename):
        tokens = np.load(filename)
        tokens = tokens.astype(np.int32)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens

    def next_batch(self):
        tokens = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.world_size
        if self.current_position + self.B * self.T * self.world_size + 1 > len(self.tokens):
            self.current_shard += 1
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = self.B * self.T * self.process_rank
        return x, y


def get_shakespeare(B, T, device=torch.device("cpu")):
    with open("shakespeare.txt") as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text[:1000])
    tokens = torch.tensor(tokens[:B * T + 1], device=device)
    x = tokens[:-1].view(B, T)
    y = tokens[1:].view(B, T)
    return x, y


class ShakespeareLoader:
    def __init__(self, B, T, process_rank, world_size):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        with open("shakespeare.txt") as f:
            text = f.read()

        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        tokens = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.world_size

        if self.current_position + self.B * self.T * self.world_size + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


class InstructionDataset(Dataset):
    def __init__(self, max_length=1024):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.dataset = load_dataset("databricks/databricks-dolly-15k")["train"]
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        instruction, context, response = item["instruction"], item["context"], item["response"]
        if context:
            prompt = f"{instruction}\nContext:\n{context}\nResponse:\n"
        else:
            prompt = f"{instruction}\nResponse:\n"

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        input_ids = prompt_ids + response_ids
        input_ids = input_ids[:self.max_length]

        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[1:]  # shift the labels to make the model predict correct things
        labels = labels[:self.max_length]

        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.eot_token] * padding_length
        labels += [-100] * (padding_length + 1)
        labels = labels[:self.max_length]  # needed to solve some bug

        attention_mask = [1] * (len(prompt_ids) + len(response_ids))
        attention_mask += [0] * padding_length
        attention_mask = attention_mask[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask)
        }

class PreferenceDataset(Dataset):
    def __init__(self, max_length=1024):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.dataset = load_dataset("andersonbcdefg/red_teaming_reward_modeling_pairwise")["train"]
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        instruction, response_a, response_b, preferred = item["prompt"], item["response_a"], item["response_b"], item["preferred"]
        positive_response = response_a if preferred == "A" else response_b
        negative_response = response_a if preferred == "B" else response_b
        positive_prompt = f"{instruction}\nResponse:\n{positive_response}"
        negative_prompt = f"{instruction}\nResponse:\n{negative_response}"

        prompt_length = len(self.tokenizer.encode(f"{instruction}\nResponse:\n"))
        positive_prompt_ids = self.tokenizer.encode(positive_prompt)
        negative_prompt_ids = self.tokenizer.encode(negative_prompt)

        positive_attention_mask = [1] * len(positive_prompt_ids)
        positive_attention_mask += [0] * max(self.max_length - len(positive_attention_mask), 0)
        positive_attention_mask = positive_attention_mask[:self.max_length]
        negative_attention_mask = [1] * len(negative_prompt_ids)
        negative_attention_mask += [0] * max(self.max_length - len(negative_attention_mask), 0)
        negative_attention_mask = negative_attention_mask[:self.max_length]

        positive_response_mask = torch.ones(size=(self.max_length,))
        positive_response_mask[:prompt_length] = 0  # mask out prompt
        positive_response_mask[len(positive_prompt_ids):] = 0  # mask out padding
        positive_response_mask = positive_response_mask[1:]  # take the elements that will be predicted
        negative_response_mask = torch.ones(size=(self.max_length,))
        negative_response_mask[:prompt_length] = 0  # mask out prompt
        negative_response_mask[len(negative_prompt_ids):] = 0  # mask out padding
        negative_response_mask = negative_response_mask[1:]  # take the elements that will be predicted

        positive_padding_length = max(self.max_length - len(positive_prompt_ids), 0)
        negative_padding_length = max(self.max_length - len(negative_prompt_ids), 0)
        positive_prompt_ids += [self.tokenizer.eot_token] * positive_padding_length
        negative_prompt_ids += [self.tokenizer.eot_token] * negative_padding_length
        positive_prompt_ids = positive_prompt_ids[:self.max_length]
        negative_prompt_ids = negative_prompt_ids[:self.max_length]

        return {
            "positive_prompt_ids": torch.tensor(positive_prompt_ids),
            "positive_attention_mask": torch.tensor(positive_attention_mask),
            "negative_prompt_ids": torch.tensor(negative_prompt_ids),
            "negative_attention_mask": torch.tensor(negative_attention_mask),
            "positive_response_mask": positive_response_mask,
            "negative_response_mask": negative_response_mask,
        }


if __name__ == "__main__":
    save_fineweb()

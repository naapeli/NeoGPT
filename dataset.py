import torch
import tiktoken


def get_shakespeare(B, T, device=torch.device("cpu")):
    with open("shakespeare.txt") as f:
        text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text[:1000])
    tokens = torch.tensor(tokens[:B * T + 1], device=device)
    x = tokens[:-1].view(B, T)
    y = tokens[1:].view(B, T)
    return x, y


# class ShakespeareLoader:
#     def __init__(self, B, T):
#         self.B = B
#         self.T = T

#         with open("shakespeare.txt") as f:
#             text = f.read()

#         tokenizer = tiktoken.get_encoding("gpt2")
#         tokens = tokenizer.encode(text[:1000])
#         self.tokens = torch.tensor(tokens[:B * T + 1])

#         self.current_position = 0

#     def next_batch(self):
#         tokens = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
#         x = tokens[:-1].view(self.B, self.T)
#         y = tokens[1:].view(self.B, self.T)
#         self.current_position += self.B * self.T

#         if self.current_position + self.B * self.T > len(self.tokens):
#             self.current_position = 0
#         return x, y

def ShakespeareLoader(B, T):
    with open("shakespeare.txt") as f:
        text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens)
    current_position = 0

    while True:
        batch_tokens = tokens[current_position:current_position + B * T + 1]
        x = batch_tokens[:-1].view(B, T)
        y = batch_tokens[1:].view(B, T)
        current_position += B * T

        if current_position + B * T > len(tokens):
            current_position = 0
        yield x, y

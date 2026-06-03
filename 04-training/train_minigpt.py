import torch
import torch.nn as nn
from torch.optim import AdamW
from minigpt import MiniGPT

import matplotlib.pyplot as plt

# --- Dataset ---
# We'll train on a tiny corpus — Shakespeare-style text
text = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to tis a consummation
""" * 50  # repeat to give model more data

# --- Tokenizer (character level for simplicity) ---
chars = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_idx[c] for c in s]
def decode(ids): return ''.join([idx_to_char[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
# print(f"Vocab size     : {vocab_size}")
# print(f"Dataset tokens : {len(data)}")
# print(f"Sample encoded : {data[:10]}")
# print(f"Sample decoded : {decode(data[:10].tolist())}")

# --- Batch Generator ---
def get_batch(data, batch_size, seq_len):
    # pick random starting positions
    ix = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i+seq_len] for i in ix])
    y = torch.stack([data[i+1 : i+seq_len+1] for i in ix])
    return x, y

# --- Model ---
SEQ_LEN    = 32
BATCH_SIZE = 16
D_MODEL    = 64
NUM_HEADS  = 4
FF_DIM     = 256
NUM_LAYERS = 4
LR         = 3e-4
EPOCHS     = 1000

model     = MiniGPT(vocab_size, D_MODEL, SEQ_LEN, NUM_HEADS, FF_DIM, NUM_LAYERS)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Training Loop ---
for epoch in range(EPOCHS):
    x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)

    logits = model(x)
    # logits: (batch, seq_len, vocab_size)
    # y:      (batch, seq_len)

    # reshape for cross entropy
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")


# --- Text Generation ---
def generate(model, prompt, max_new_tokens=100):
    model.eval()
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long)
    
    generated = list(input_ids[0].numpy())
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to seq_len if too long
            input_crop = torch.tensor([generated[-SEQ_LEN:]], dtype=torch.long)
            
            logits = model(input_crop)
            
            # take logits at last position only
            next_logits = logits[0, -1, :]
            
            # sample from distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
    
    return decode(generated)

# --- Test Generation ---
print("\n--- Generated Text ---")
print(generate(model, prompt="to be", max_new_tokens=200))


# --- Loss Curve ---
losses = []

for epoch in range(EPOCHS):
    x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MiniGPT Training Loss")
plt.savefig("loss_curve.png")
print("Loss curve saved.")
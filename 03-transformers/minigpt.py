import torch
import torch.nn as nn
from transformers import AutoTokenizer

# --- config ---
vocab_size  = 1000
d_model     = 64
max_seq_len = 50
head_dim    = 16

# --- Step 1 ---
class TokenAndPositionalEmbedding(nn.Module):
    """Combines learned token embeddings with learned positional embeddings."""
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return self.token_emb(x) + self.pos_emb(positions)

# --- Step 2 ---
class SingleHeadAttention(nn.Module):
    """Scaled dot-product attention with causal mask for a single head."""
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.scale = head_dim ** 0.5

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        return output, weights


# --- Step 3 ---

class MultiHeadAttention(nn.Module):
    """Runs attention across multiple heads in parallel and projects output."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, self.head_dim)
            for _ in range(num_heads)
        ])

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        head_outputs = [head(x)[0] for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat)
        return output


# --- Step 4 ---

class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation applied position-wise."""
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


# --- Step 5 ---

class TransformerBlock(nn.Module):
    """One transformer layer: multi-head attention and feed-forward with residual connections and LayerNorm."""
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff        = FeedForward(d_model, ff_dim)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x


# --- Step 6 ---

class MiniGPT(nn.Module):
    """GPT-style language model stacking transformer blocks over token and positional embeddings."""
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(vocab_size, d_model, max_seq_len)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)       # (batch, seq_len, d_model)
        x = self.blocks(x)          # (batch, seq_len, d_model)
        x = self.norm(x)            # (batch, seq_len, d_model)
        logits = self.out_proj(x)   # (batch, seq_len, vocab_size)
        return logits


# --- step 7 ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentences = [
    "the transformer architecture changed everything",
    "attention is all you need",
]

encoded = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    max_length=50,
    return_tensors="pt"
)

input_ids = encoded["input_ids"]
print(f"Tokenized shape : {input_ids.shape}")

model = MiniGPT(
    vocab_size  = tokenizer.vocab_size,
    d_model     = 64,
    max_seq_len = 50,
    num_heads   = 4,
    ff_dim      = 256,
    num_layers  = 4
)

logits = model(input_ids)
predicted_ids = logits.argmax(dim=-1)
predicted_tokens = [tokenizer.decode(ids) for ids in predicted_ids]

print(f"Logits shape      : {logits.shape}")
print(f"Predicted shape   : {predicted_ids.shape}")
print(f"\nPredicted tokens:")
for i, tokens in enumerate(predicted_tokens):
    print(f"  Sentence {i+1}: {tokens}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

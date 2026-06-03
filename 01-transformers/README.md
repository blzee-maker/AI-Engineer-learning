# Session 01 — MiniGPT: Transformer Architecture from Scratch

**Status:** Complete  
**Stack:** Python · PyTorch · HuggingFace Tokenizers

---

## What it does

Implements a GPT-style transformer language model from scratch in a single file — no shortcuts, no black boxes. Every component is built and named to match the original "Attention Is All You Need" paper:

| Step | Component | What it is |
|------|-----------|------------|
| 1 | `TokenAndPositionalEmbedding` | Converts token IDs into vectors; adds learned position info |
| 2 | `SingleHeadAttention` | Scaled dot-product attention with causal mask so the model can't see future tokens |
| 3 | `MultiHeadAttention` | Runs N attention heads in parallel, then merges them |
| 4 | `FeedForward` | Two-layer MLP with GELU — the "think" step after attention |
| 5 | `TransformerBlock` | Stacks attention + FFN with residual connections and LayerNorm |
| 6 | `MiniGPT` | Stacks N transformer blocks into a complete language model |

The demo tokenizes real sentences with BERT's tokenizer and runs a full forward pass, producing logits over the entire vocabulary (30,522 tokens).

```
Tokenized shape : torch.Size([2, 11])
Logits shape    : torch.Size([2, 11, 30522])
Total parameters: 3,987,034
```

## What I learned

- **Attention is just matrix math.** Q×Kᵀ scores how much each token should attend to every other token. Softmax turns scores into weights. Weights×V produces the output.
- **The causal mask is what makes it autoregressive.** An upper-triangular mask of `-inf` before softmax ensures position `t` can only see positions `≤ t`.
- **Multi-head attention = run it N times in parallel.** Each head learns different relationships (syntax, coreference, proximity). Their outputs are concatenated and projected back.
- **Residual connections are load-bearing.** Without `x + attention(x)`, gradients vanish in deep networks. The residual is the highway.
- **LayerNorm stabilizes training.** Applied before each sub-layer (Pre-LN), it keeps activations in a sane range regardless of depth.
- **Parameter count scales fast.** A 4-layer, 64-dim model already has ~4M parameters.

## How to run it

```bash
pip install torch transformers
python minigpt.py
```

No GPU required — the model is small enough to run on CPU in under a second.

## Model config

```
d_model     = 64
num_heads   = 4
ff_dim      = 256
num_layers  = 4
max_seq_len = 50
vocab_size  = 30,522  (bert-base-uncased)
```

## Architecture diagram

```
Input token IDs
      │
      ▼
TokenAndPositionalEmbedding
      │
      ▼
TransformerBlock × 4
  ├── LayerNorm
  ├── MultiHeadAttention (causal mask)
  │     └── SingleHeadAttention × 4
  ├── Residual
  ├── LayerNorm
  ├── FeedForward (Linear → GELU → Linear)
  └── Residual
      │
      ▼
LayerNorm → Linear → Logits (vocab_size)
```

## Reference

Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

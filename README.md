# MiniGPT — Transformer Built from Scratch

A minimal GPT-style transformer implemented in PyTorch, built step-by-step to make every component of the architecture transparent and easy to follow.

## What's in the file

The code builds the transformer piece by piece, mirroring how the architecture is typically taught:

| Step | Class | Description |
|------|-------|-------------|
| 1 | `TokenAndPositionalEmbedding` | Combines token embeddings with learned positional embeddings |
| 2 | `SingleHeadAttention` | Scaled dot-product attention with causal (autoregressive) mask |
| 3 | `MultiHeadAttention` | Runs multiple attention heads in parallel and projects their output |
| 4 | `FeedForward` | Two-layer MLP with GELU activation |
| 5 | `TransformerBlock` | Assembles attention + FFN with residual connections and LayerNorm |
| 6 | `MiniGPT` | Stacks N transformer blocks into a full language model |
| 7 | Demo | Tokenizes real sentences with BERT tokenizer and runs a forward pass |

## Model config (default)

```
d_model     = 64
num_heads   = 4
ff_dim      = 256
num_layers  = 4
max_seq_len = 50
vocab_size  = bert-base-uncased vocab (~30k)
```

## Requirements

```bash
pip install torch transformers
```

## Run it

```bash
python transformer.py
```

Sample output:

```
Tokenized shape : torch.Size([2, 11])
Logits shape    : torch.Size([2, 11, 30522])
Predicted shape : torch.Size([2, 11])
Total parameters: 3,987,034
```

## Architecture overview

```
Input token IDs
      │
      ▼
TokenAndPositionalEmbedding
      │
      ▼
TransformerBlock × N
  ├── MultiHeadAttention (causal mask)
  │     └── SingleHeadAttention × num_heads
  ├── Residual + LayerNorm
  ├── FeedForward (Linear → GELU → Linear)
  └── Residual + LayerNorm
      │
      ▼
LayerNorm → Linear → Logits (vocab_size)
```

## Purpose

This is a learning-oriented implementation. The goal is readability over performance — each class maps directly to a named concept in the original paper so you can follow along line by line.

## Reference

Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

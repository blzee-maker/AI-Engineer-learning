import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "Hello, world!",
    "The transformer architecture changed everything.",
    "tiktoken is OpenAI's fast BPE tokenizer.",
    "Tokenization splits text into subword units.",
]

print("=== BPE Tokenization with tiktoken ===\n")

for text in texts:
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"Text   : {text}")
    print(f"IDs    : {tokens}")
    print(f"Tokens : {decoded}")
    print(f"Count  : {len(tokens)}\n")

# common words are single tokens; rare/long words split into subwords
words = ["cat", "cats", "caterpillar", "supercalifragilistic", "AI", "GPT", "transformer"]
print("=== Word-level token counts ===")
for word in words:
    ids = enc.encode(word)
    pieces = [enc.decode([t]) for t in ids]
    print(f"  {word:<25} → {len(ids)} token(s): {pieces}")

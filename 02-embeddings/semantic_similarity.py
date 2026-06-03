from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "A dog is running in the park.",
    "A puppy is playing outside.",
    "The stock market crashed today.",
    "Financial markets saw a sharp decline.",
    "Cats are independent animals.",
    "The weather is nice today.",
]

print("=== Semantic Similarity with Sentence Transformers ===\n")

embeddings = model.encode(sentences, convert_to_tensor=True)
print(f"Embedding shape: {embeddings.shape}\n")

similarity = F.cosine_similarity(
    embeddings.unsqueeze(1),
    embeddings.unsqueeze(0),
    dim=-1,
)

print("All pairs ranked by similarity:")
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append((similarity[i, j].item(), sentences[i], sentences[j]))

pairs.sort(reverse=True)
for score, s1, s2 in pairs:
    print(f"  [{score:.3f}]  \"{s1}\"")
    print(f"           \"{s2}\"")
    print()

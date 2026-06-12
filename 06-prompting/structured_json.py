import ollama
import json

# Step 1 — naive: just ask. Run several passages and watch it occasionally misbehave.
SYSTEM_NAIVE = """You are a literary tone analyst for an audiobook pipeline.
Classify the passage using one of the labels: wistful, menacing, tender, detached, frantic.
Respond in JSON with keys "tone" and "confidence" (0 to 1)."""

def classify_naive(passage: str) -> str:
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_NAIVE},
            {"role": "user", "content": f'Passage: "{passage}"'},
        ],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()

# Step 2 — hardened: Ollama's format="json" + tight constraints + parse safely.
SYSTEM_STRICT = """You are a literary tone analyst for an audiobook pipeline.
Classify the passage using EXACTLY one of the labels: wistful, menacing, tender, detached, frantic.

Output ONLY a JSON object with these keys:
- "tone": one of the five labels, lowercase
- "confidence": a number between 0 and 1

No markdown, no code fences, no explanation. JSON only."""

def classify_json(passage: str) -> dict:
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_STRICT},
            {"role": "user", "content": f'Passage: "{passage}"'},
        ],
        format="json",          # Ollama's JSON-mode: constrains output to valid JSON
        options={"temperature": 0},
    )
    raw = resp["message"]["content"].strip()
    try:
        data = json.loads(raw)          # should never throw if format="json" held
        return data
    except json.JSONDecodeError:
        return {"error": "invalid json", "raw": raw}

passages = [
    "The room was empty now, the way rooms are after everyone leaves.",
    "He smiled, but his hand never left the knife on the table.",
    "Run! she screamed, dragging the child toward the door.",
]

print("=== NAIVE ===")
for p in passages:
    print(classify_naive(p))

print("\n=== HARDENED (parsed) ===")
for p in passages:
    result = classify_json(p)
    print(result, "| tone:", result.get("tone"))
import ollama
import json

SYSTEM_STRICT = """You are a literary tone analyst for an audiobook pipeline.
Classify the passage using EXACTLY one of the labels: wistful, menacing, tender, detached, frantic.

Output ONLY a JSON object with these keys:
- "tone": one of the five labels, lowercase
- "confidence": a number between 0 and 1

No markdown, no code fences, no explanation. JSON only."""

def classify_prefill(passage: str) -> dict:
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_STRICT},
            {"role": "user", "content": f'Passage: "{passage}"'},
            {"role": "assistant", "content": "{"},   # PREFILL: seed the JSON opening
        ],
        options={"temperature": 0},
    )
    # The model continues AFTER our "{", so prepend it back before parsing
    raw = "{" + resp["message"]["content"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid json", "raw": raw}

passages = [
    "The room was empty now, the way rooms are after everyone leaves.",
    "He smiled, but his hand never left the knife on the table.",
    "Run! she screamed, dragging the child toward the door.",
]

print("=== PREFILL ===")
for p in passages:
    result = classify_prefill(p)
    print(result, "| tone:", result.get("tone"))
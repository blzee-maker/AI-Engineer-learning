# 05-api-mechanics/api_mechanics_streaming.py
from google import genai
from google.genai import types
import ollama

from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
PROMPT = "Describe a quiet seaside town at dawn in three sentences."

# ============================================================
# PART 1 — STREAMING
# ============================================================

print("=== GEMINI (streaming) ===")
# generate_content_stream returns an ITERATOR of chunks, not one response.
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
):
    # end="" and flush=True so tokens appear inline as they arrive
    print(chunk.text, end="", flush=True)
print("\n")

print("=== OLLAMA (streaming) ===")
# stream=True turns the response into a generator of partial chunks
for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)
print("\n")

# ============================================================
# PART 2 — TOKEN COUNTING
# ============================================================

# for gemini
print("=== TOKEN COUNTING (Gemini) ===")

# Count BEFORE sending — for cost/context estimation
pre = client.models.count_tokens(model="gemini-2.5-flash", contents=PROMPT)
print("Input tokens (counted before sending):", pre.total_tokens)

# Count AFTER — actual usage from the response metadata
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
u = resp.usage_metadata
print("Prompt tokens:", u.prompt_token_count)
print("Output tokens:", u.candidates_token_count)
print("Total tokens:", u.total_token_count)

# for ollama - token counting
print("\n=== TOKEN COUNTING (Ollama) ===")

# Ollama has no count_tokens endpoint — the counts come back in the
# response metadata after generation (non-streaming call here).
resp = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": PROMPT}],
)
print("Prompt tokens:", resp["prompt_eval_count"])
print("Output tokens:", resp["eval_count"])
print("Total tokens:", resp["prompt_eval_count"] + resp["eval_count"])

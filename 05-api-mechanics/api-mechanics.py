# ----- GEMINI (cloud) -----
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()  # loads GEMINI_API_KEY from .env file into environment variables

client = genai.Client()  # reads GEMINI_API_KEY from env

def gemini_call(prompt, temperature, max_tokens):
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,    
            max_output_tokens=max_tokens,
            thinking_config = types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text

# ----- OLLAMA (local) -----
import ollama

def ollama_call(prompt, temperature, max_tokens):
    resp = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],  # the message LIST
        options={
            "temperature": temperature,
            "num_predict": max_tokens,   # Ollama's name for max output tokens
        },
    )
    return resp["message"]["content"]

# ============================================================
# THE EXPERIMENT — run each twice, observe
# ============================================================
PROMPT = "In one sentence, describe a wistful character reflecting on an old letter."

print("--- temp=0, run 1 ---"); print(ollama_call(PROMPT, 0.0, 60))
print("--- temp=0, run 2 ---"); print(ollama_call(PROMPT, 0.0, 60))
print("--- temp=1, run 1 ---"); print(ollama_call(PROMPT, 1.0, 60))
print("--- temp=1, run 2 ---"); print(ollama_call(PROMPT, 1.0, 60))
print("--- max_tokens=15 (does it truncate?) ---"); print(ollama_call(PROMPT, 0.0, 15))



print("--- temp=0, run 1 ---"); print(gemini_call(PROMPT, 0.0, 60))
print("--- temp=0, run 2 ---"); print(gemini_call(PROMPT, 0.0, 60))
print("--- temp=1, run 1 ---"); print(gemini_call(PROMPT, 1.0, 60))
print("--- temp=1, run 2 ---"); print(gemini_call(PROMPT, 1.0, 60))
print("--- max_tokens=15 (does it truncate?) ---"); print(gemini_call(PROMPT, 0.0, 15))
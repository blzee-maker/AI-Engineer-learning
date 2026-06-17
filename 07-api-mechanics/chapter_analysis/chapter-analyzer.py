# =============================================================
# Week 3 · CAPSTONE: Manuscript Chapter Analyzer
# Integrates the whole week into one working pipeline slice.
#
#   [1] token counting   
#   [2] dialogue tool    
#   [3] model tone/summary call
#   [4] Pydantic schema validation
#   [5] ONE repair retry on invalid output
#   [6] provider-swappable: Gemini OR llama  
#
# Run:  python chapter_analyzer.py
# =============================================================

import json
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
load_dotenv()


# -------------------------------------------------------------
# THE CONTRACT
# This schema is what the rest of the pipeline is allowed to trust.
# If the model's output doesn't fit this shape, it does NOT pass.
# -------------------------------------------------------------
class ChapterAnalysis(BaseModel):
    tone: str = Field(description="One-word dominant tone, e.g. melancholic, tense, hopeful")
    summary: str = Field(description="One sentence summary of the chapter")
    dialogue_count: int = Field(description="Number of dialogue segments (filled by our code, not the model)")


# -------------------------------------------------------------
# [2] DETERMINISTIC TOOL
# Exact, free, reliable. The model never does this.
# -------------------------------------------------------------

import re
def count_dialogue_segments(text: str) -> int:
    """Count quoted dialogue segments. A proxy — see README caveats."""
    return len(re.findall(r'"[^"]*"', text))


# -------------------------------------------------------------
# PROVIDER LAYER
# Same job, two backends. The ONLY thing that changes downstream
# is which function we call. This is the hybrid-routing idea in code.
# -------------------------------------------------------------

def call_gemini(prompt: str) -> tuple[str, dict]:
    from google import genai
    from google.genai import types
    client = genai.Client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",   # ask Gemini for JSON directly
        ),
    )
    u = resp.usage_metadata
    usage = {"input": u.prompt_token_count, "output": u.candidates_token_count}
    return resp.text, usage


def call_llama(prompt: str) -> tuple[str, dict]:
    import ollama
    resp = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        format="json",   # Ollama's JSON mode — constrains output
    )
    usage = {"input": resp["prompt_eval_count"], "output": resp["eval_count"]}
    return resp["message"]["content"], usage


PROVIDERS = {"gemini": call_gemini, "llama": call_llama}


# -------------------------------------------------------------
# PROMPT BUILDER
# Note: we ONLY ask the model for what it's good at (tone, summary).
# We do NOT ask it to count dialogue — our code does that.
# -------------------------------------------------------------

def build_prompt(chapter: str) -> str:
    return (
        "You are a literary analysis assistant. Analyze the chapter below.\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        '  "tone": one-word dominant tone\n'
        '  "summary": one sentence summarizing the chapter\n'
        "No markdown, no backticks, no extra text.\n\n"
        f"CHAPTER:\n{chapter}"
    )


# -------------------------------------------------------------
# VALIDATION + REPAIR  ([4] and [5])
# This is the production answer to Tuesday's malformed-output bug.
# Try to validate. If it fails, hand the error BACK to the model once.
# -------------------------------------------------------------

def parse_and_validate(raw: str, dialogue_count: int) -> ChapterAnalysis:
    data = json.loads(raw)                 # may raise if not JSON
    data["dialogue_count"] = dialogue_count  # WE fill this, not the model
    return ChapterAnalysis(**data)         # may raise ValidationError


def analyze_chapter(chapter: str, provider: str = "gemini") -> dict:
    call = PROVIDERS[provider]
    dialogue_count = count_dialogue_segments(chapter)   # [2] deterministic first

    prompt = build_prompt(chapter)
    raw, usage = call(prompt)                           # [3] model call
    total_usage = dict(usage)

    # [4] try to validate
    try:
        result = parse_and_validate(raw, dialogue_count)
    except (json.JSONDecodeError, ValidationError) as err:
        # [5] ONE repair retry: show the model its own broken output + the error
        print(f"  ⚠ validation failed ({type(err).__name__}); attempting repair retry...")
        repair_prompt = (
            f"{prompt}\n\nYour previous reply could not be parsed.\n"
            f"Error: {err}\nReturn corrected JSON only."
        )
        raw2, usage2 = call(repair_prompt)
        total_usage["input"] += usage2["input"]
        total_usage["output"] += usage2["output"]
        result = parse_and_validate(raw2, dialogue_count)  # if THIS fails, we let it raise

    # [6] assemble final report, including cost transparency
    return {
        "provider": provider,
        "analysis": result.model_dump(),
        "tokens": total_usage,
    }


# -------------------------------------------------------------
# DEMO
# -------------------------------------------------------------

if __name__ == "__main__":
    CHAPTER = ('''Mara stood at the harbor's edge as the last light drained from the sky.\n'
        '"You\'re leaving again," the boy said.\n'
        'She didn\'t answer at first. "The sea doesn\'t ask permission," she finally replied.'
        'The boy watched her boots sink into the wet sand, saying nothing.''')

    for prov in ("gemini", "llama"):
        print(f"\n=== {prov.upper()} ===")
        try:
            report = analyze_chapter(CHAPTER, provider=prov)
            print(json.dumps(report, indent=2))
        except Exception as e:
            print(f"  FAILED after retry: {type(e).__name__}: {e}")
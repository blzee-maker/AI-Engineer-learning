"""Loads a prompt config from prompts/, formats it into messages, runs it on Ollama.

The point of this project: the prompt is DATA (a YAML file), the code is a
generic engine. Adding a new task = adding a YAML file + a schema entry,
with zero changes to this loader.
"""

import json
from pathlib import Path

import ollama
import yaml

from schemas import validate

# Resolve relative to this file, so the loader works from any working directory.
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> dict:
    path = PROMPTS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_messages(cfg: dict, passage: str) -> list:
    messages = [{"role": "system", "content": cfg["system"]}]
    # few-shot examples, if declared
    for ex in cfg.get("examples", []):
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": f'Passage: "{passage}"'})
    # prefill: a trailing assistant message the model continues from
    if cfg.get("prefill"):
        messages.append({"role": "assistant", "content": cfg["prefill"]})
    return messages


def run_prompt(name: str, passage: str) -> dict:
    cfg = load_prompt(name)
    messages = build_messages(cfg, passage)

    # Two hardening strategies, picked per prompt:
    #   prefill declared -> seed the JSON opening ourselves
    #   no prefill       -> Ollama's JSON mode constrains the whole output
    prefill = cfg.get("prefill", "")
    extra = {} if prefill else {"format": "json"}

    resp = ollama.chat(
        model=cfg["model"],
        messages=messages,
        options={"temperature": cfg.get("temperature", 0)},
        **extra,
    )
    raw = resp["message"]["content"].strip()
    # Some models continue after the prefill, some repeat it — handle both.
    if prefill and not raw.startswith(prefill):
        raw = prefill + raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid json", "raw": raw}

    # semantic check — structure != correctness
    problems = validate(name, data)
    if problems:
        return {"error": "; ".join(problems), "raw": data}
    return data


if __name__ == "__main__":
    # One passage, five tasks — the same engine runs them all.
    passage = (
        "'We leave at dawn,' Mira whispered, glancing at the door. "
        "'The Tchaikovsky manuscript stays here,' Saoirse replied, "
        "her hand resting on the revolver in her coat."
    )
    tasks = [
        "character_extraction",
        "dialogue_detection",
        "speaker_attribution",
        "pronunciation_flagging",
        "tone_classification",
    ]
    print(f'Passage: "{passage}"\n')
    for task in tasks:
        print(f"=== {task} ===")
        print(json.dumps(run_prompt(task, passage), indent=2))
        print()

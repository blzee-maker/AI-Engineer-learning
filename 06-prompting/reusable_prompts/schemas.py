"""Expected JSON output shape per task.

A schema here is just: required key -> allowed Python type(s).
json.loads already gives us Python objects, so type-checking the parsed
dict is the simplest validation possible — no extra libraries needed.

Two layers of checking:
  1. Structural — are the right keys there, with the right types?
  2. Semantic  — are the VALUES sensible? (structure != correctness:
     {"tone": "banana", "confidence": 7} is valid JSON with valid keys
     and still completely wrong.)
"""

SCHEMAS = {
    "tone_classification": {
        "tone": str,
        "confidence": (int, float),
    },
    "character_extraction": {
        "characters": list,
    },
    "dialogue_detection": {
        "has_dialogue": bool,
        "quotes": list,
    },
    "speaker_attribution": {
        "attributions": list,
    },
    "pronunciation_flagging": {
        "flags": list,
    },
}

# The closed label set for tone_classification — anything else is a hallucinated label.
TONE_LABELS = {"wistful", "menacing", "tender", "detached", "frantic"}

# For tasks whose value is a list of objects: which keys each item must have.
ITEM_KEYS = {
    "character_extraction": ("characters", {"name"}),
    "speaker_attribution": ("attributions", {"quote", "speaker"}),
    "pronunciation_flagging": ("flags", {"word", "reason", "hint"}),
}


def validate(task: str, data: dict) -> list:
    """Check parsed model output against the task's schema.

    Returns a list of problems; an empty list means the output is valid.
    """
    schema = SCHEMAS.get(task)
    if schema is None:
        return [f"no schema registered for task: {task}"]

    problems = []

    # 1. Structural: required keys and their types
    for key, expected in schema.items():
        if key not in data:
            problems.append(f"missing key: {key}")
        elif not isinstance(data[key], expected):
            problems.append(f"key '{key}' should be {expected}, got {type(data[key]).__name__}")

    # 2. Semantic: values that are typed correctly but still wrong
    if task == "tone_classification":
        if isinstance(data.get("tone"), str) and data["tone"] not in TONE_LABELS:
            problems.append(f"tone '{data['tone']}' is not in the allowed labels {sorted(TONE_LABELS)}")
        conf = data.get("confidence")
        if isinstance(conf, (int, float)) and not 0 <= conf <= 1:
            problems.append(f"confidence {conf} is outside [0, 1]")

    if task in ITEM_KEYS:
        list_key, required_item_keys = ITEM_KEYS[task]
        for item in data.get(list_key, []):
            if not isinstance(item, dict) or not required_item_keys <= set(item):
                problems.append(f"item {item!r} in '{list_key}' is missing keys {required_item_keys}")

    return problems

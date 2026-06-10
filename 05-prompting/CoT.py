# Chain Of Thought (CoT) prompting

import ollama 

# Only CoT

# SYSTEM = """You are a literary tone analyst for an audiobook pipeline.
# classify each passage using only these labels: wistful, menacing, tender, detached, frantic.

# for each passage, first reason step by step about the imagery, word choice, and emotional distance.
# Then on a final line, output exactly one label: Tone : <label>"""


# CoT with few-shot examples that pin the wistful/detached boundary

SYSTEM = """You are a literary tone analyst for an audiobook pipeline.
Classify passages using ONLY these labels: wistful, menacing, tender, detached, frantic.

wistful = gentle longing for something gone
detached = emotionally distant, observational, no personal investment

First reason briefly about imagery, word choice, and emotional distance.
Then output exactly: Tone: <label>"""

# Few-shot examples that pin the wistful/detached boundary
EXAMPLES = """Passage: "He noted the time of death, closed the file, and moved on."
Reasoning: Clinical, procedural, no emotional investment.
Tone: detached

Passage: "The room was empty now, but she could still hear the laughter that used to fill it."
Reasoning: Describes absence but reaches back toward what was lost — longing.
Tone: wistful"""


def classify_tone(passage: str) -> str:
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'{EXAMPLES}\nPassage: "{passage}"\nTone:'},
        ],
        options = {"temperature":0},
    )

    return resp["message"]["content"].strip()

test = "The room was empty now, the way rooms are after everyone leaves."
print(classify_tone(test))
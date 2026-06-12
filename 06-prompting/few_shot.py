import ollama

SYSTEM = """You are a literary tone analyst for an audiobook pipeline.
Classify each passage using only these labels: wistful, menacing, tender, detached, frantic.
Output exactly one label, lowercase, nothing else."""

EXAMPLES = """
Passage: "He smiled, but his hand never left the knife on the table.
Tone: menacing

Passage: "He tucked the blanked under her chin and stayed until her breathing slowed.
Tone: tender

Passage: "He noted the time of death, closed the file, and moved to the next case."
Tone: detached

Passage: "The room was empty now, but she could still hear the laughter that used to fill it."
Tone: wistful

"""

def classify_tone(passage: str) -> str:
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'{EXAMPLES}\n\nPassage: "{passage}"\nTone'},
        ],
        options={"temperature":0},
    )
    return resp["message"]["content"].strip()

test = "The room was empty now, the way rooms are after everyone leaves."
print(classify_tone(test))


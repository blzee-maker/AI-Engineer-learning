import ollama

# User role only, no system prompt, no examples

def classify_tone_user_role(passage: str) -> str:
    instruction = """You are a literary tone analyst for an audiobook pipeline.
Classify passages using ONLY these labels: wistful, menacing, tender, detached, frantic.
Output exactly: Tone: <label>"""
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": f'{instruction}\n\nPassage: "{passage}"'},
        ],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()



# System prompt only, no examples

def classify_tone_system_role(passage: str) -> str:
    SYSTEM = """You are a literary tone analyst for an audiobook pipeline.
Classify passages using ONLY these labels: wistful, menacing, tender, detached, frantic.
Output exactly: Tone: <label>"""
    resp = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'Passage: "{passage}"\nTone:'},
        ],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()


# For system or user role, no examples Options [1: user role only, 2: system role only, 3: both roles]

def classify_tone(passage: str, option: int) -> str:
    if option == 1:
        return classify_tone_user_role(passage)
    elif option == 2:
        return classify_tone_system_role(passage)
    elif option == 3:
        # Combine both system and user role
        SYSTEM = """You are a literary tone analyst for an audiobook pipeline.
        Classify passages using ONLY these labels: wistful, menacing, tender, detached, frantic.
        Output exactly: Tone: <label>"""
        
        resp = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f'Passage: "{passage}"\nTone:'},
            ],
            options={"temperature": 0},
        )
        return resp["message"]["content"].strip()
    
test = "The room was empty now, the way rooms are after everyone leaves."
print("User role only:", classify_tone(test, option=1))
print("System role only:", classify_tone(test, option=2))
print("Both roles:", classify_tone(test, option=3))
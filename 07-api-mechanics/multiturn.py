# 05-api-mechanics/api_mechanics_multiturn.py
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

# THE conversation — a list YOU own. Starts empty.
history = []

def chat_turn(user_text):
    # 1. append the user's message to the history
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    # 2. send the ENTIRE history every time (model is stateless)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,   # <-- the whole list, not just the new message
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    # 3. append the model's reply so the next turn can see it
    reply = resp.candidates[0].content
    history.append(reply)

    # observe how the history (and thus token cost) grows each turn
    print(f"[history now has {len(history)} messages]")
    print("MODEL:", resp.text, "\n")

# --- a 3-turn conversation that REQUIRES memory ---
chat_turn("I'm designing a character named Mara. She's a retired sailor.")
chat_turn("What kind of voice should the narrator use for her?")   # 'her' = needs turn 1
chat_turn("Now give her a single line of dialogue in that voice.") # needs turns 1 & 2
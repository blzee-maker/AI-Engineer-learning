from google import genai
from google.genai import types

import re

from dotenv import load_dotenv
load_dotenv()  # loads GEMINI_API_KEY from .env file into environment variables

client = genai.Client()

# Stage 1: define the actual funtion

def count_dialogue_lines(ch_text: str) -> int:
    """Counts the number of dialogue lines in a given text. 
    A dialogue line is defined as lines with a double-quote."""
    return len(re.findall(r'"[^"]*"', ch_text))

# Stage 2: describe the tool to the model
# the model reads this decription and schema. it never sees the code above.

tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="count_dialogue_lines",
            description="Counts the number of dialogue lines in a chapter of text provided.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ch_text": types.Schema(type="STRING", description="The full text of the chapter."),
                },
                required=["ch_text"],
            ),
        )
    ]
)

CHAPTER = '''"Hello, how are you?" said Alice. "I'm sure he is fine!" said jack. 
"I'm good, thanks!" replied Bob.
"The sun was setting over the horizon," Alice thought to herself.'''

# Stage 3: Send the message with the tools available

messages = [types.Content(role="user", parts=[types.Part(
    text=f"How many dialogue lines are in this chapter?\n\n{CHAPTER}"
)])]

# AUTO mode (the default) lets the model decide whether to call the tool
# or answer from its own knowledge. ANY would force a tool call every time.
config = types.GenerateContentConfig(
    tools=[tool],
    system_instruction=(
        "You have a tool for counting dialogue lines. Use it only when the "
        "question is about counting dialogue. For anything else, answer "
        "directly from your own knowledge without calling the tool."
    ),
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
    ),
)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=config,
)

# Stage 4: Did the model ask to call the tool?
# Scan every part — the model may return text alongside a function call,
# so we can't assume parts[0] is the call.

def find_function_call(response):
    for part in response.candidates[0].content.parts:
        if part.function_call:
            return part.function_call
    return None

call = find_function_call(resp)

if call:
    print("MODEL REQUESTED:", call.name, dict(call.args))

    # Stage 5: The code runs the funtion
    result = count_dialogue_lines(**call.args)
    print("Your CODE COMPUTED", result)

    # Stage 6: Append the models request and the result, send back
    messages.append(resp.candidates[0].content)
    messages.append(types.Content(role="user", parts=[types.Part(
        function_response=types.FunctionResponse(
            name = call.name,
            response={"result": result},
        )
    )]))

    final = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=config,
    )
    print("Final Answer: ", final.text)
else:
    print("Model answered directly (no tool call): ", resp.text)

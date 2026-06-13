import re
import json
import ollama


# -----------------------------
# Tool implementation
# -----------------------------
def count_dialogue_lines(ch_text: str) -> int:
    """
    Counts quoted dialogue segments.

    Example:
    "Hello"
    "How are you?"

    Returns 2.
    """
    return len(re.findall(r'"([^"]*)"', ch_text))


# -----------------------------
# Tool schema for Ollama
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "count_dialogue_lines",
            "description": "Counts the number of dialogue segments enclosed in double quotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ch_text": {
                        "type": "string",
                        "description": "The chapter text to analyze."
                    }
                },
                "required": ["ch_text"]
            }
        }
    }
]


CHAPTER = '''"Hello, how are you?" said Alice.
"I'm sure he is fine!" said Jack.
"I'm good, thanks!" replied Bob.
"The sun was setting over the horizon," Alice thought to herself.'''


# -----------------------------
# Ask model
# -----------------------------
response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": f"What is the tone of this chapter?\n\n{CHAPTER}" # Switched with Question: How many dialogue lines are in this chapter?, To see how the model reacts when the tool is not able to handle the query.
        }
    ],
    tools=tools
)


# -----------------------------
# Did model request a tool?
# -----------------------------
message = response["message"]

if message.get("tool_calls"):

    tool_call = message["tool_calls"][0]

    function_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]

    print("MODEL REQUESTED:", function_name)
    print("ARGS:", arguments)

    # Execute tool — use the authoritative local text, not the model's
    # reconstructed copy (the model drops/mangles quotes when retyping it).
    result = count_dialogue_lines(CHAPTER)

    print("TOOL RESULT:", result)

    # Send result back
    final_response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"How many dialogue lines are in this chapter?\n\n{CHAPTER}"
            },
            message,
            {
                "role": "tool",
                "name": function_name,
                "content": str(result)
            }
        ]
    )

    print("\nFINAL ANSWER:")
    print(final_response["message"]["content"])

else:
    print("MODEL ANSWERED DIRECTLY:")
    print(message["content"])
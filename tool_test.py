import ollama

# 1. Describe the tool to the LLM (it reads this like a manual)
tools = [
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set the cabin target temperature in Celsius and turn on the heater.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_c": {
                        "type": "integer",
                        "description": "Target temperature in Celsius, between 16 and 30",
                    }
                },
                "required": ["temp_c"],
            },
        },
    }
]

# 2. Ask the LLM, and give it the tool list
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "You are an in-car voice assistant. Use tools to control the car. Answer in 1-2 short sentences."},
        {"role": "user", "content": "I'm freezing"},
    ],
    tools=tools,
    options={"temperature": 0},
)

# 3. See what the LLM decided
print("Text answer:", response.message.content)
print("Tool calls:", response.message.tool_calls)
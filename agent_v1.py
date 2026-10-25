import ollama
from vehicle import car, set_temperature

tools = [
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set the cabin target temperature in Celsius and turn on the heater.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_c": {"type": "integer", "description": "Target temperature in Celsius, between 16 and 30"}
                },
                "required": ["temp_c"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are an in-car voice assistant. Use tools to control the car. Answer in 1-2 short sentences."},
    {"role": "user", "content": "I'm freezing"},
]

print("Car BEFORE:", car)

# STEP A: LLM decides
response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=tools, options={"temperature": 0})
messages.append(response.message)

# STEP B: Our code runs the tool the LLM asked for
for call in response.message.tool_calls or []:
    name = call.function.name
    args = call.function.arguments
    print("LLM asks for:", name, args)

    if name == "set_temperature":
        result = set_temperature(**args)
    else:
        result = {"ok": False, "error": "Unknown tool"}

    print("Car result:", result)
    messages.append({"role": "tool", "content": str(result), "tool_name": name})

# STEP C: LLM reads the real result and answers the driver
final = ollama.chat(model="qwen2.5:7b", messages=messages, tools=tools, options={"temperature": 0})

print("Assistant says:", final.message.content)
print("Car AFTER:", car)
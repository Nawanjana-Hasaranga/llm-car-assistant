import ollama
from vehicle import car, get_vehicle_status, set_temperature, set_seat_heating, set_windows

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are an in-car voice assistant. Use tools to read or control the car. "
    "Never guess vehicle data: use get_vehicle_status to check it. "
    "Never say you changed something unless you called a tool and it returned ok: True. "
    "If a request is not possible, say so and do not pretend to act. "
    "Answer in 1-2 short sentences."
)


# ---------- 1. Tool descriptions (what the LLM reads) ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_status",
            "description": "Get the car's current status: battery %, range (how far the car can still drive in km), cabin and target temperature, heater, seat heating, windows, speed. Use this whenever the driver asks about the car's condition.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "set_seat_heating",
            "description": "Set the driver's seat heating level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "0 = off, 1 = low, 2 = medium, 3 = high"}
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_windows",
            "description": "Open or close the car windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "open_windows": {"type": "boolean", "description": "true to open, false to close"}
                },
                "required": ["open_windows"],
            },
        },
    },
]

# ---------- 2. Tool name -> real Python function ----------
functions = {
    "get_vehicle_status": get_vehicle_status,
    "set_temperature": set_temperature,
    "set_seat_heating": set_seat_heating,
    "set_windows": set_windows,
}


# ---------- 3. The agent loop ----------
def ask_assistant(user_text, messages):
    messages.append({"role": "user", "content": user_text})

    for step in range(5):  # max 5 rounds, so it can never loop forever
        response = ollama.chat(model=MODEL, messages=messages, tools=tools, options={"temperature": 0})
        messages.append(response.message)

        # Case 2: no tool needed -> this is the final answer
        if not response.message.tool_calls:
            return response.message.content

        # Case 1: run every tool the LLM asked for
        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments or {}
            print(f"   [tool]   {name} {args}")

            if name in functions:
                try:
                    result = functions[name](**args)
                except Exception as error:
                    result = {"ok": False, "error": str(error)}
            else:
                result = {"ok": False, "error": "Unknown tool"}

            print(f"   [result] {result}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Sorry, I couldn't complete that."


# ---------- 4. The chat ----------
if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Car assistant ready. Type 'quit' to exit.")

    while True:
        user_text = input("\nYou: ")
        if user_text.lower() == "quit":
            break
        answer = ask_assistant(user_text, messages)
        print("Car:", answer)
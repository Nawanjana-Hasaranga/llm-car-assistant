import re
import ollama
from vehicle import car, get_vehicle_status, set_temperature, adjust_temperature, set_seat_heating, set_windows

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are an in-car voice assistant. Use tools to read or control the car. "
    "For ANY question about the car's current state (heater, temperature, seat heating, windows, battery, range, speed), "
    "always call get_vehicle_status first. Never answer from memory. "
    "Never say you changed something unless you called a tool and it returned ok: True. "
    "If a request is not possible, say so and do not pretend to act. "
    "When a request is not possible, offer ONE specific alternative with an exact value "
    "(for example: 'Would you like 30 degrees instead?'). "
    "If the driver says yes, only do exactly the action you proposed. "
    "Never invent a value the driver did not say or confirm. "
    "Answer in 1-2 short sentences."
)

# Patterns that mean "I am doing / did an action" (built from real failures)
CLAIM_PATTERNS = [
    r"\bi(?:'ve| have|'ll| will|'m| am)?\s+(?:now\s+|just\s+|already\s+)?(?:set|setting|increas\w*|decreas\w*|lower\w*|rais\w*|turn\w*|open\w*|clos\w*|adjust\w*|activat\w*|switch\w*|chang\w*|reduc\w*|boost\w*|made|making)\b",
    r"^\s*(?:setting|increasing|decreasing|lowering|raising|turning|opening|closing|adjusting|switching|changing|reducing)\b",
    r"\b(?:has|have) been (?:set|increased|decreased|lowered|raised|turned|opened|closed|adjusted|changed|reduced|activated|switched)\b",
    r"\b(?:is|are) now (?:set|on|off|open|closed|at|warmer|cooler)\b",
    r"\b(?:is|are) (?:now )?turned (?:up|down)\b",
]

ACTION_TOOLS = ["set_temperature", "adjust_temperature", "set_seat_heating", "set_windows"]


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
            "description": "Set the cabin target temperature to an exact value in Celsius. Use ONLY when the driver says a specific number.",
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
            "name": "adjust_temperature",
            "description": "Make the cabin warmer or cooler relative to the current target temperature. Use this when the driver describes a feeling (cold, freezing, chilly, hot, boiling, warm) or says warmer/cooler, without a specific number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "change_c": {"type": "integer", "description": "Degrees to change. Positive = warmer (e.g. 2), negative = cooler (e.g. -2). Use about 3 for strong feelings like freezing or boiling, 1-2 for mild ones."}
                },
                "required": ["change_c"],
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
    "adjust_temperature": adjust_temperature,
    "set_seat_heating": set_seat_heating,
    "set_windows": set_windows,
}


# ---------- 3. Guardrail: does the answer claim an action? ----------
def claims_action(answer):
    # Split into sentences; questions ("Would you like me to...?") are not claims
    parts = re.split(r"(?<=[.!?;])\s+", answer.lower().strip())
    statements = [part for part in parts if not part.endswith("?")]
    return any(re.search(pattern, sentence) for sentence in statements for pattern in CLAIM_PATTERNS)


# ---------- 4. The agent loop ----------
def ask_assistant(user_text, messages):
    messages.append({"role": "user", "content": user_text})
    action_done = False  # did an action tool succeed during this request?

    for step in range(5):  # max 5 rounds, so it can never loop forever
        response = ollama.chat(model=MODEL, messages=messages, tools=tools, options={"temperature": 0})
        messages.append(response.message)

        # Case 2: no tool call -> candidate final answer
        if not response.message.tool_calls:
            answer = response.message.content

            # Guardrail: claimed an action, but nothing happened in the car
            if claims_action(answer) and not action_done:
                print("   [guard]  Answer claims an action but no tool succeeded. Asking the model to correct itself.")
                if step == 4:
                    return "Sorry, I couldn't complete that. Nothing was changed."
                messages.append({
                    "role": "system",
                    "content": "Check failed: you described an action, but you did not call a tool, so NOTHING changed in the car. "
                               "Either call the correct tool now for what the driver asked, "
                               "or tell the driver what is possible and ask them to confirm. Do not claim actions.",
                })
                continue

            return answer

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

            if name in ACTION_TOOLS and isinstance(result, dict) and result.get("ok"):
                action_done = True

            print(f"   [result] {result}")
            messages.append({"role": "tool", "content": str(result), "tool_name": name})

    return "Sorry, I couldn't complete that."


# ---------- 5. The chat ----------
if __name__ == "__main__":
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Car assistant ready. Type 'quit' to exit.")

    while True:
        user_text = input("\nYou: ")
        if user_text.lower() == "quit":
            break
        answer = ask_assistant(user_text, messages)
        print("Car:", answer)
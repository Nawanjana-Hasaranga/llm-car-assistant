import time
import csv
import ollama
from agent import MODEL, SYSTEM_PROMPT, tools

# ---------- The test set: sentence + correct tool ----------
# "expected" = list of acceptable tools. None = no tool should be called.
# "args" (optional) = values the tool must receive.
test_cases = [
    # Reading status
    {"input": "What's my battery level?", "expected": ["get_vehicle_status"]},
    {"input": "How far can I drive?", "expected": ["get_vehicle_status"]},
    {"input": "Is the heater on?", "expected": ["get_vehicle_status"]},
    {"input": "Are the windows open?", "expected": ["get_vehicle_status"]},

    # Temperature
    {"input": "Set the temperature to 21 degrees", "expected": ["set_temperature"], "args": {"temp_c": 21}},
    {"input": "Make it 24 degrees in here", "expected": ["set_temperature"], "args": {"temp_c": 24}},
    {"input": "I'm freezing", "expected": ["set_temperature", "set_seat_heating"]},
    {"input": "It's boiling in here", "expected": ["set_temperature"]},
    {"input": "I'm a bit chilly", "expected": ["set_temperature", "set_seat_heating"]},

    # Seat heating
    {"input": "Turn on the seat heating", "expected": ["set_seat_heating"]},
    {"input": "Seat heating to medium", "expected": ["set_seat_heating"], "args": {"level": 2}},
    {"input": "Make my seat really warm", "expected": ["set_seat_heating"], "args": {"level": 3}},
    {"input": "Turn off the seat heating", "expected": ["set_seat_heating"], "args": {"level": 0}},

    # Windows
    {"input": "Open the windows", "expected": ["set_windows"], "args": {"open_windows": True}},
    {"input": "Close the windows", "expected": ["set_windows"], "args": {"open_windows": False}},
    {"input": "Let some fresh air in", "expected": ["set_windows"], "args": {"open_windows": True}},

    # Impossible or unsafe -> should NOT call a tool
    {"input": "Set the temperature to 50 degrees", "expected": [None]},
    {"input": "Order me a pizza", "expected": [None]},
    {"input": "Play some music", "expected": [None]},
    {"input": "What's the capital of France?", "expected": [None]},
]


# ---------- Ask the LLM once and see which tool it chooses ----------
def get_first_decision(user_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    response = ollama.chat(model=MODEL, messages=messages, tools=tools, options={"temperature": 0})
    calls = response.message.tool_calls or []
    if not calls:
        return None, {}
    return calls[0].function.name, calls[0].function.arguments or {}


# ---------- Run all tests ----------
results = []
passed = 0
total_time = 0

print(f"Running {len(test_cases)} tests with {MODEL}...\n")

for i, case in enumerate(test_cases, start=1):
    start = time.time()
    tool, args = get_first_decision(case["input"])
    seconds = time.time() - start
    total_time += seconds

    correct = tool in case["expected"]
    if correct and "args" in case:
        for key, value in case["args"].items():
            if args.get(key) != value:
                correct = False

    if correct:
        passed += 1
    status = "PASS" if correct else "FAIL"

    print(f"{i:2}. {status} | {case['input']:<38} | got: {tool} {args} | {seconds:.1f}s")
    results.append([case["input"], case["expected"], tool, args, status, round(seconds, 2)])

# ---------- Summary ----------
accuracy = passed / len(test_cases) * 100
print(f"\nAccuracy: {passed}/{len(test_cases)} = {accuracy:.0f}%")
print(f"Average time per request: {total_time / len(test_cases):.1f} s")

# ---------- Save results to a file ----------
with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["input", "expected", "got_tool", "got_args", "status", "seconds"])
    writer.writerows(results)
print("Results saved to eval_results.csv")
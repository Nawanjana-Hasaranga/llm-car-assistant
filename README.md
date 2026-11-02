# LLM Car Assistant

A local, privacy-friendly **in-car assistant** that understands natural language ("I'm freezing", "Let some fresh air in") and safely controls a simulated vehicle through **LLM tool calling**.

> 🚧 **Status: in development.** This is a self-learning project. The text-based agent and its evaluation are working; voice input/output and owner's-manual Q&A are next on the roadmap.

---

## The problem

Modern cars move climate, seat and window controls into touchscreens and menus, which pulls the driver's eyes off the road. Classic voice assistants only understand fixed commands such as "Temperature 22 degrees", so natural requests like *"I'm freezing"* fail.

**Goal:** an assistant that understands what the driver means, performs the right action, never pretends to do something it didn't do, and runs fully offline on local hardware.

## What it does (current version)

- **Understands natural language:** "It's boiling in here" → makes the cabin 3 °C cooler; "Make my seat really warm" → seat heating level 3.
- **Reads real vehicle data:** battery, range, temperatures, heater, seat heating, windows, speed. No invented values.
- **Acts through tools, not text:** the LLM only *requests* an action; Python code executes it.
- **Safety enforced in code:** temperature limited to 16–30 °C, seat heating 0–3, windows cannot open above 100 km/h. Unsafe requests are blocked even if the model asks for them.
- **Honest responses:** if a request is impossible it says so and offers an alternative instead of pretending.
- **Multi-turn memory:** remembers the conversation, so a follow-up "yes" confirms the previous suggestion.

### Example session (real output)

```
You: What's my battery level?
   [tool]   get_vehicle_status {}
Car: Your battery level is at 68%.

You: Set the temperature to 50 degrees
Car: The requested temperature is out of the allowed range (16-30 degrees Celsius).
     Would you like to set the temperature to 30 degrees instead?

You: yes
   [tool]   set_temperature {'temp_c': 30}
   [result] {'ok': True, 'target_temp_c': 30}
Car: The temperature has been set to 30 degrees Celsius.

You: Order me a pizza
Car: I can't place orders for you. You can place the order using your phone or another device.
```

## How it works

```
Driver (text)  ──►  LLM agent (Qwen2.5-7B, local)  ──►  tool call request
                          ▲                                   │
                          │                                   ▼
                 tool result (ok / error)  ◄──  Python executes tool + safety checks
                          │                                   │
                          ▼                                   ▼
                 honest answer to driver            simulated vehicle state
```

The agent loop runs up to 5 rounds per request: the model either answers directly or asks for one or more tools, the results are fed back, and the model writes the final answer from real data.

### Tools available to the model

| Tool | Purpose |
|---|---|
| `get_vehicle_status` | Read battery, range, temperatures, heater, seat heating, windows, speed |
| `set_temperature` | Set an exact target temperature (driver names a number) |
| `adjust_temperature` | Warmer/cooler relative to the current target (driver describes a feeling) |
| `set_seat_heating` | Seat heating level 0–3 |
| `set_windows` | Open or close the windows |

## Models and stack

| Component | Used |
|---|---|
| LLM | **Qwen2.5-7B-Instruct** (`qwen2.5:7b`) |
| Runtime | **Ollama**, fully local and offline |
| Decoding | temperature = 0 (deterministic tool selection) |
| Language | Python 3.13, `ollama` Python library |
| Speech-to-text (planned) | OpenAI Whisper |

## Run it

**Requirements:** Python 3.11+, [Ollama](https://ollama.com), about 8 GB free RAM for the 7B model.

```bash
git clone https://github.com/Nawanjana-Hasaranga/llm-car-assistant.git
cd llm-car-assistant

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
ollama pull qwen2.5:7b
```

| Command | What it does |
|---|---|
| `python agent.py` | Interactive chat with the car assistant (type `quit` to exit) |
| `python evaluate.py` | Runs the evaluation and writes `eval_results.csv` |
| `python test_vehicle.py` | Unit test of the simulated vehicle and its safety rules |

## Evaluation

`evaluate.py` runs **22 test sentences** with a fresh conversation each time and checks the model's first decision:

1. **Right tool?** (or no tool, for requests that should be declined)
2. **Exact values correct?** e.g. "21 degrees" → `temp_c = 21`
3. **Do the values make sense?** e.g. "It's boiling" must make the cabin *cooler* (`change_c < 0`)

### Results (qwen2.5:7b, local laptop)

| Category | Tests | Passed |
|---|---|---|
| Reading vehicle status | 4 | 4 |
| Exact temperature | 2 | 2 |
| Temperature from feelings (warmer/cooler) | 5 | 5 |
| Seat heating | 4 | 4 |
| Windows | 3 | 3 |
| Should decline (unsafe / out of scope) | 4 | 4 |
| **Total** | **22** | **22 (100%)** |

**Latency:** about **0.4 s** per decision once the model is loaded (average 0.6 s including the first, cold-start request of about 5 s).

### How the score got there

| Version | Change | Score | Finding |
|---|---|---|---|
| v1 | First test set (20 tests) | 95% | "How far can I drive?" not linked to *range* |
| v2 | Clearer tool description | 100% | Misleading: "It's boiling" set 22 °C, making the car **warmer** |
| v3 | Added value checks to tests | 95% | Test now catches the bug |
| v4 | New relative `adjust_temperature` tool, 22 tests | 95% | Boiling fixed; **regression** on "Is the heater on?" |
| v5 | More specific system prompt | **100%** | All tests pass, no regressions |

**Lessons:** a high score can hide wrong behaviour unless the tests check values, not just tool names; good tool design (relative warmer/cooler, like real car buttons) works better than asking the model to guess a number; and every prompt or tool change needs a full re-run to catch regressions.

> **Limitation:** 100% applies to this 22-sentence test set only. A larger and more varied test set is on the roadmap.

## Safety design

The LLM is **not** treated as a safety mechanism. Two layers are used:

1. **Schema and prompt:** the model is told the limits, which avoids most bad requests.
2. **Code checks in `vehicle.py`:** always enforced. In testing, with limits removed from the schema, the model requested 50 °C; the code rejected it and the model explained the limit to the driver.

## Project structure

```
vehicle.py        Simulated vehicle state, actions and safety rules
agent.py          Tool schemas, agent loop and interactive chat
evaluate.py       Automated evaluation (22 test cases)
eval_results.csv  Latest evaluation results
test_vehicle.py   Unit tests for the vehicle and safety rules
chat_test.py      Early experiments: system prompt and temperature
tool_test.py      Early experiment: first tool call
agent_v1.py       Early experiment: single tool-call cycle
```

## Roadmap

- [x] Local LLM setup and prompt experiments
- [x] Simulated vehicle with safety rules
- [x] Tool-calling agent with multi-turn memory
- [x] Automated evaluation with value checks
- [ ] Voice: Whisper speech-to-text and text-to-speech
- [ ] RAG: answer questions from a real owner's manual (e.g. warning lights)
- [ ] Larger test set and comparison of model sizes (3B vs 7B vs 8B)
- [ ] LoRA fine-tuning of a small model on car commands
- [ ] Simple dashboard showing the vehicle state live

## Author

**Nawanjana Haththellage**, MSc Autonomy Technologies, FAU Erlangen-Nürnberg
from agent import claims_action

# Real answers from our tests: should the guard catch them?
test_answers = [
    # Fake actions (no tool was called) -> guard must catch them
    ("I've decreased the temperature by 5 degrees Celsius to make it more comfortable.", True),
    ("Let's make the cabin warmer. Setting the temperature to 22°C.", True),
    ("Setting the seat heating to high.", True),
    ("I'm sorry, but the temperature must be between 16 and 30°C. Setting it to the closest allowed temperature, which is 16°C.", True),
    ("I'll set it to the maximum, 30°C, instead.", True),
    ("The heating is turned up; is that helping?", True),

    # Honest answers -> guard must NOT block them
    ("Your battery level is at 68%.", False),
    ("I'm here to help. How can I assist you further?", False),
    ("The temperature must be between 16 and 30 degrees Celsius, so I can't set it to 15.", False),
    ("I can't place orders for you. You can place the order using your phone or another device.", False),
    ("The maximum temperature I can set is 30 degrees Celsius. Would you like to set the temperature to 30 degrees instead?", False),
]

passed = 0
for answer, expected in test_answers:
    result = claims_action(answer)
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    print(f"{status} | expected {str(expected):<5} got {str(result):<5} | {answer[:60]}")

print(f"\nGuard tests: {passed}/{len(test_answers)}")
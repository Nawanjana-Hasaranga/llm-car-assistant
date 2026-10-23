import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "You are an in-car voice assistant. Answer in 1-2 short sentences."},
        {"role": "user", "content": "I'm freezing"},
    ],
)

print(response.message.content)
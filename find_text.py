import json

with open("manuals/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

while True:
    search = input("\nText to find (or 'quit'): ").lower()
    if search == "quit":
        break

    found = 0
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        position = text.lower().find(search)
        if position != -1:
            found += 1
            snippet = text[max(0, position - 80):position + 160]
            print(f"chunk {i} | PDF page {chunk['page']} | ...{snippet}...")
    print(f"{found} chunks contain '{search}'")
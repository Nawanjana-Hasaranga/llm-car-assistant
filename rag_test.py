import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"
TOP_K = 3              # how many sections to give the LLM
MIN_SCORE = 0.55       # below this, the manual probably doesn't cover the question

# Practice text (simplified, written for learning; NOT from a real manual)
manual_sections = [
    "Engine warning light (orange): the engine management system has detected a fault. You may continue driving at moderate speed. Have the vehicle checked by a workshop soon.",
    "Engine warning light flashing: engine misfiring that can damage the catalytic converter. Reduce speed and have the vehicle checked immediately.",
    "Tyre pressure warning light: the pressure in one or more tyres is too low. Stop when safe and check the tyre pressures.",
    "Recommended tyre pressures are shown on the label on the driver's door frame.",
    "Brake system warning light (red): brake fluid level too low or brake system fault. Stop as soon as it is safe and do not continue driving.",
    "Charging: connect the charging cable to the socket on the left rear side of the vehicle. Charging starts automatically.",
]

RAG_PROMPT = (
    "You are an in-car assistant. Answer the driver's question using ONLY the manual excerpts provided. "
    "If the excerpts do not contain the answer, say exactly: 'The manual doesn't cover that.' "
    "If the question can be answered with yes or no, start with the correct 'Yes' or 'No' that matches the manual. "
    "If the manual says to stop or not drive, say that first and clearly. "
    "Answer in 1-2 short sentences."
)


def embed(texts):
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(response.embeddings)


section_vectors = embed(["search_document: " + text for text in manual_sections])


# ---------- 1. RETRIEVE ----------
def search_manual(question):
    question_vector = embed(["search_query: " + question])[0]
    scores = section_vectors @ question_vector / (
        np.linalg.norm(section_vectors, axis=1) * np.linalg.norm(question_vector)
    )
    ranking = np.argsort(scores)[::-1][:TOP_K]
    return [(scores[i], manual_sections[i]) for i in ranking]


# ---------- 2. AUGMENT + 3. GENERATE ----------
def answer_with_manual(question):
    results = search_manual(question)
    best_score = results[0][0]

    print("  Found sections:")
    for score, text in results:
        print(f"    {score:.2f} | {text[:70]}...")

    if best_score < MIN_SCORE:
        return "The manual doesn't cover that."

    excerpts = "\n".join(f"[{n}] {text}" for n, (score, text) in enumerate(results, start=1))
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_PROMPT},
            {"role": "user", "content": f"Manual excerpts:\n{excerpts}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content


def answer_from_memory(question):
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an in-car assistant. Answer in 1-2 short sentences."},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},
    )
    return response.message.content


# ---------- Compare: without manual vs with manual ----------
while True:
    question = input("\nQuestion (or 'quit'): ")
    if question.lower() == "quit":
        break
    print("WITHOUT manual:", answer_from_memory(question))
    print("WITH manual:   ", answer_with_manual(question))
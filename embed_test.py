import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"

# Practice text (simplified, written for learning; NOT from a real manual)
manual_sections = [
    "Engine warning light (orange): the engine management system has detected a fault. You may continue driving at moderate speed. Have the vehicle checked by a workshop soon.",
    "Engine warning light flashing: engine misfiring that can damage the catalytic converter. Reduce speed and have the vehicle checked immediately.",
    "Tyre pressure warning light: the pressure in one or more tyres is too low. Stop when safe and check the tyre pressures.",
    "Recommended tyre pressures are shown on the label on the driver's door frame.",
    "Brake system warning light (red): brake fluid level too low or brake system fault. Stop as soon as it is safe and do not continue driving.",
    "Charging: connect the charging cable to the socket on the left rear side of the vehicle. Charging starts automatically.",
]


def embed(texts):
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(response.embeddings)


# 1. Turn every manual section into a vector (done once)
print("Embedding manual sections...")
section_vectors = embed(["search_document: " + text for text in manual_sections])
print(f"{len(manual_sections)} sections, each vector has {section_vectors.shape[1]} numbers\n")

# 2. Ask questions and find the most similar sections
while True:
    question = input("Question (or 'quit'): ")
    if question.lower() == "quit":
        break

    question_vector = embed(["search_query: " + question])[0]

    # Cosine similarity: 1.0 = same meaning, 0 = unrelated
    scores = section_vectors @ question_vector / (
        np.linalg.norm(section_vectors, axis=1) * np.linalg.norm(question_vector)
    )

    ranking = np.argsort(scores)[::-1]  # highest score first
    for rank, i in enumerate(ranking[:3], start=1):
        print(f"  {rank}. score {scores[i]:.2f} | {manual_sections[i][:75]}...")
    print()
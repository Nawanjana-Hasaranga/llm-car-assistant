import json
import re
import numpy as np
import ollama
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"
TOP_K = 4            # chunks given to the LLM
CANDIDATES = 20      # how many results each search method contributes
MIN_SCORE = 0.65     # tuned on the real manual: answers 0.78-0.85, junk 0.57-0.61

# ---------- Load the saved index ----------
vectors = np.load("manuals/vectors.npy")
with open("manuals/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
vector_norms = np.linalg.norm(vectors, axis=1)

# ---------- Keyword index (BM25) ----------
STOPWORDS = {
    "a", "an", "the", "is", "are", "i", "my", "me", "can", "do", "does", "what", "how", "if",
    "there", "it", "to", "of", "on", "in", "and", "or", "should", "with", "this", "that",
    "be", "for", "you", "your", "when", "which", "where", "keep",
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
print(f"Loaded {len(chunks)} chunks from the VW Tiguan manual (meaning + keyword index)")

RAG_PROMPT = (
    "You are an in-car assistant for a VW Tiguan. Answer the driver's question using ONLY the manual excerpts provided.\n"
    "Follow these rules in order:\n"
    "1. If NONE of the excerpts is relevant to the question, reply exactly: 'The manual doesn't cover that.' and nothing else.\n"
    "2. Otherwise, answer from the relevant excerpts. If the question is vague, give the most likely meaning based on the excerpts.\n"
    "3. If the manual says to stop or not drive on, say that first.\n"
    "4. If the question can be answered with yes or no, start with the correct 'Yes' or 'No'.\n"
    "5. Answer in 1-3 short sentences and end with the PDF page label you used, like (PDF page 19). "
    "Never cite page numbers that appear inside the excerpt text."
)


# ---------- 1. RETRIEVE (hybrid) ----------
def search_manual(question):
    # Meaning search
    question_vector = np.array(ollama.embed(model=EMBED_MODEL, input=["search_query: " + question]).embeddings[0])
    vector_scores = vectors @ question_vector / (vector_norms * np.linalg.norm(question_vector))

    # Keyword search
    keyword_scores = bm25.get_scores(tokenize(question))

    # Top candidates from each method
    vector_top = np.argsort(vector_scores)[::-1][:CANDIDATES]
    keyword_top = np.argsort(keyword_scores)[::-1][:CANDIDATES]

    # Reciprocal Rank Fusion: points for a good position in each list
    fused = {}
    for rank, i in enumerate(vector_top):
        fused[i] = fused.get(i, 0) + 1 / (60 + rank)
    for rank, i in enumerate(keyword_top):
        fused[i] = fused.get(i, 0) + 1 / (60 + rank)

    best = sorted(fused, key=fused.get, reverse=True)[:TOP_K]
    results = []
    for i in best:
        source = ("V" if i in vector_top else "") + ("K" if i in keyword_top else "")
        results.append((source, float(vector_scores[i]), float(keyword_scores[i]), chunks[i]))
    return results, float(vector_scores.max())


# ---------- 2. AUGMENT + 3. GENERATE ----------
def answer_question(question):
    results, best_meaning_score = search_manual(question)

    print("  Found (V = meaning search, K = keyword search):")
    for source, v_score, k_score, chunk in results:
        print(f"    {source:<2} | meaning {v_score:.2f} | keyword {k_score:5.1f} | PDF page {chunk['page']:>3} | {chunk['text'][:50]}...")

    if best_meaning_score < MIN_SCORE:
        return "The manual doesn't cover that."

    excerpts = "\n\n".join(f"[PDF page {chunk['page']}] {chunk['text']}" for _, _, _, chunk in results)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_PROMPT},
            {"role": "user", "content": f"Manual excerpts:\n{excerpts}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content


if __name__ == "__main__":
    while True:
        question = input("\nQuestion (or 'quit'): ")
        if question.lower() == "quit":
            break
        print("Answer:", answer_question(question))
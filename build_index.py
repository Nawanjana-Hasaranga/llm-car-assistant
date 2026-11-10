import json
import re
import time
import numpy as np
import ollama
from pypdf import PdfReader

PDF_PATH = "manuals/vw_tiguan.pdf"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000   # characters per chunk
OVERLAP = 200       # characters shared between neighbouring chunks
BATCH_SIZE = 32     # chunks embedded per request

# ---------- 1. Extract text page by page ----------
print("Reading PDF...")
reader = PdfReader(PDF_PATH)
print(f"{len(reader.pages)} pages")

chunks = []
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = re.sub(r"\s+", " ", text).strip()  # clean up spaces and line breaks

    # ---------- 2. Cut the page into overlapping chunks ----------
    start = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        if len(piece) > 100:  # skip tiny leftovers
            chunks.append({"page": page_number, "text": piece})
        start += CHUNK_SIZE - OVERLAP

print(f"{len(chunks)} chunks created")

# ---------- 3. Embed all chunks (in batches) ----------
print("Embedding chunks (this takes a few minutes)...")
start_time = time.time()
vectors = []
for i in range(0, len(chunks), BATCH_SIZE):
    batch = ["search_document: " + chunk["text"] for chunk in chunks[i:i + BATCH_SIZE]]
    response = ollama.embed(model=EMBED_MODEL, input=batch)
    vectors.extend(response.embeddings)
    print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

print(f"Done in {time.time() - start_time:.0f} s")

# ---------- 4. Save the index ----------
np.save("manuals/vectors.npy", np.array(vectors, dtype=np.float32))
with open("manuals/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)
print("Saved manuals/vectors.npy and manuals/chunks.json")

# ---------- Quick look at two chunks ----------
for chunk in chunks[40:42]:
    print(f"\n--- page {chunk['page']} ---\n{chunk['text'][:300]}...")
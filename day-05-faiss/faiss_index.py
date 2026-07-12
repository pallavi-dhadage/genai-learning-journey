import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.")

# 2. Document corpus
documents = [
    "Tesla reported record quarterly deliveries of 484,507 vehicles, beating estimates.",
    "The company announced a new battery factory in Nevada and a price cut in China.",
    "Shares rose 5% in after-hours trading.",
    "Apple's iPhone revenue grew 6% year-over-year to $69.7 billion.",
    "Microsoft's cloud business Azure grew 30% this quarter.",
    "Amazon's advertising revenue jumped 24% to $14.7 billion."
]

print(f"📄 Loaded {len(documents)} documents.")

# 3. Embed documents
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')
print(f"🔢 Embedding shape: {embeddings.shape}")

# 4. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors.")

# 5. Save index and metadata
faiss.write_index(index, "day-05-faiss/faiss_index.bin")
with open("day-05-faiss/documents.pkl", "wb") as f:
    pickle.dump(documents, f)
print("💾 Index and documents saved to disk.")

# 6. Load back from disk
def load_index_and_docs():
    index_loaded = faiss.read_index("day-05-faiss/faiss_index.bin")
    with open("day-05-faiss/documents.pkl", "rb") as f:
        docs_loaded = pickle.load(f)
    return index_loaded, docs_loaded

index_loaded, docs_loaded = load_index_and_docs()
print("📂 Index and documents reloaded from disk.")

# 7. Search function
def search_faiss(query: str, top_k: int = 3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index_loaded.search(query_embedding, top_k)
    results = [(docs_loaded[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# 8. Interactive search loop
print("\n🔍 FAISS Semantic Search Engine")
print("Type your query (or 'exit' to quit):")

while True:
    query = input("\n> ")
    if query.lower() in ["exit", "quit"]:
        break
    results = search_faiss(query, top_k=3)
    print("\n📌 Top results (by Euclidean distance, lower = closer):")
    for i, (doc, dist) in enumerate(results, 1):
        print(f"{i}. {doc}")
        print(f"   (Distance: {dist:.4f})")

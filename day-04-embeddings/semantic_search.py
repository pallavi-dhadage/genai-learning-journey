import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the embedding model (small & fast)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.")

# 2. Our document corpus (financial snippets)
documents = [
    "Tesla reported record quarterly deliveries of 484,507 vehicles, beating estimates.",
    "The company announced a new battery factory in Nevada and a price cut in China.",
    "Shares rose 5% in after-hours trading.",
    "Apple's iPhone revenue grew 6% year-over-year to $69.7 billion.",
    "Microsoft's cloud business Azure grew 30% this quarter.",
    "Amazon's advertising revenue jumped 24% to $14.7 billion."
]

print(f"📄 Loaded {len(documents)} documents.")

# 3. Embed all documents (compute once)
doc_embeddings = model.encode(documents)
print(f"🔢 Embedding shape: {doc_embeddings.shape}")

# 4. Search function
def search(query: str, top_k: int = 3) -> list:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]

# 5. Interactive loop
print("\n🔍 Semantic Search Engine")
print("Type your query (or 'exit' to quit):")

while True:
    query = input("\n> ")
    if query.lower() in ["exit", "quit"]:
        break
    results = search(query, top_k=3)
    print("\n📌 Top results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {doc}")
        print(f"   (Similarity: {score:.4f})\n")

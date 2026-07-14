import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# =============================================
# 1. Load environment variables (GROQ_API_KEY)
# =============================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =============================================
# 2. Load embedding model and FAISS index
# =============================================
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded.")

# Load FAISS index and documents from Day 5
index = faiss.read_index("day-05-faiss/faiss_index.bin")
with open("day-05-faiss/documents.pkl", "rb") as f:
    documents = pickle.load(f)
print(f"📂 Loaded FAISS index with {index.ntotal} vectors and {len(documents)} documents.")

# =============================================
# 3. Retrieval function
# =============================================
def retrieve(query: str, top_k: int = 3) -> list:
    """
    Embed the query and return the top_k most similar documents (as text strings).
    """
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# =============================================
# 4. RAG generation function (with streaming)
# =============================================
def rag_answer(query: str, top_k: int = 3) -> str:
    # 4a. Retrieve relevant documents
    docs = retrieve(query, top_k)
    context = "\n\n".join(docs)
    
    # 4b. Build the prompt
    prompt = f"""You are a helpful financial assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, say that you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""
    
    # 4c. Call Groq with streaming
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True
    )
    
    # 4d. Collect and print streamed tokens
    full_response = ""
    print("Assistant: ", end="", flush=True)
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta
    print("\n")
    return full_response

# =============================================
# 5. Interactive loop
# =============================================
print("\n🤖 RAG Chatbot (retrieval + generation)")
print("Type your question (or 'exit' to quit):")

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    rag_answer(query, top_k=3)

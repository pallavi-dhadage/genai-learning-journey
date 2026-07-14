import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("day-05-faiss/faiss_index.bin")
with open("day-05-faiss/documents.pkl", "rb") as f:
    documents = pickle.load(f)
print(f"📂 Loaded FAISS index with {index.ntotal} vectors and {len(documents)} documents.")

def retrieve(query: str, top_k: int = 3) -> list:
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def rag_answer(query: str, top_k: int = 3) -> tuple:
    docs = retrieve(query, top_k)
    context = "\n\n".join(docs)
    prompt = f"""You are a financial assistant. Use the context to answer. If you don't know, say so.

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content, docs

def evaluate_answer(query: str, answer: str, context: str) -> dict:
    """Ask Groq to score the answer on faithfulness and relevancy (1-5)."""
    prompt = f"""You are an expert evaluator. Given the following context, answer, and question, rate:
- Faithfulness (1-5): Does the answer stick to the facts in the context? (5 = fully grounded, 1 = hallucinated)
- Relevancy (1-5): Does the answer directly address the question? (5 = perfect, 1 = off-topic)

Return only a JSON like: {{"faithfulness": X, "relevancy": Y}}

Context: {context[:1000]}
Question: {query}
Answer: {answer}

Ratings:"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # deterministic
    )
    try:
        import json
        ratings = json.loads(response.choices[0].message.content)
        return ratings
    except:
        return {"faithfulness": 0, "relevancy": 0}

# Test questions
test_questions = [
    "What were Tesla's delivery numbers?",
    "How did Apple's iPhone revenue perform?",
    "What did Microsoft's cloud business do?",
    "What happened to Amazon's advertising revenue?",
    "Which company announced a new battery factory?",
    "What was the after-hours stock movement mentioned?"
]

print(f"🧪 Evaluating {len(test_questions)} answers...")
scores = []
for q in test_questions:
    answer, docs = rag_answer(q, top_k=3)
    context = "\n\n".join(docs)
    ratings = evaluate_answer(q, answer, context)
    scores.append(ratings)
    print(f"Q: {q[:30]}... -> Faith: {ratings.get('faithfulness',0)} | Rel: {ratings.get('relevancy',0)}")

# Average scores
faith_avg = np.mean([s.get('faithfulness', 0) for s in scores])
rel_avg = np.mean([s.get('relevancy', 0) for s in scores])
print(f"\n📊 Average Faithfulness: {faith_avg:.2f}/5")
print(f"📊 Average Relevancy:    {rel_avg:.2f}/5")

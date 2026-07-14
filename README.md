# 🚀 GenAI Learning Journey

A structured, 30‑day, project‑based deep dive into Generative AI Engineering.  
Each day includes a working Python script, a practical concept, and a commit to track real progress.

> **Current Status:** Days 1–6 complete (Foundations → RAG Pipeline).

---

## 📚 Progress Log

| Day | Topic | What I Built |
| :---: | :--- | :--- |
| 01 | Prompt Engineering | News summariser using Groq (LLaMA 3.3). |
| 02 | Streaming Responses | Interactive CLI chat that streams tokens in real time. |
| 03 | Function Calling | Agent with stock price & calculator tools. |
| 04 | Semantic Search | Document retrieval using `sentence-transformers` & cosine similarity. |
| 05 | Vector Databases (FAISS) | Persistent FAISS index with save/load & fast similarity search. |
| 06 | **RAG Pipeline** | Combined FAISS retrieval + Groq generation to answer questions grounded in documents. |

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **APIs / Frameworks:** Groq, Sentence-Transformers, FAISS, scikit-learn, NumPy  
- **Package Management:** Poetry  
- **Version Control:** Git & GitHub

---

## 🏃 Quick Start

```bash
# Clone the repo
git clone https://github.com/pallavi-dhadage/genai-learning-journey.git
cd genai-learning-journey

# Install dependencies (using Poetry)
poetry install

# Run a specific day (e.g., Day 6 - RAG Pipeline)
poetry run python day-06-rag/rag_pipeline.py
Note: Set up your API keys (e.g., GROQ_API_KEY) in a .env file at the root.

🗓️ Roadmap (Next Steps)
Day 7: Evaluate RAG with RAGAS (faithfulness, relevance)

Day 8: LangGraph – State Machines

Day 9: Agent with Memory

Day 10: Multi‑Tool Agent

... and more until Day 30 (Capstone: Financial Co‑Pilot).

📝 License
MIT – feel free to use this to kick‑start your own GenAI journey!

# 🚀 GenAI Learning Journey

A structured, 30‑day, project‑based deep dive into Generative AI Engineering.  
Each day includes a working Python script, a practical concept, and a commit to track real progress.

> **Current Status:** Days 1–8 complete (Foundations → RAG → Agentic Workflows).

---

## 📚 Progress Log

| Day | Topic | What I Built |
| :---: | :--- | :--- |
| 01 | Prompt Engineering | News summariser using Groq (LLaMA 3.3). |
| 02 | Streaming Responses | Interactive CLI chat that streams tokens in real time. |
| 03 | Function Calling | Agent with stock price & calculator tools. |
| 04 | Semantic Search | Document retrieval using `sentence-transformers` & cosine similarity. |
| 05 | Vector Databases (FAISS) | Persistent FAISS index with save/load & fast similarity search. |
| 06 | RAG Pipeline | Combined FAISS retrieval + Groq generation to answer questions grounded in documents. |
| 07 | RAG Evaluation | Manual evaluation using Groq as a judge (faithfulness & relevancy scoring). |
| 08 | **LangGraph Agent** | State machine agent with tool calling, conversation memory, and dynamic routing. |

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **LLMs & APIs** | Groq (LLaMA 3.3), OpenAI (optional) |
| **Embeddings & Retrieval** | Sentence-Transformers, FAISS, scikit-learn |
| **Agent Frameworks** | LangGraph, LangChain Tools |
| **Evaluation** | Manual prompt‑based scoring (Groq as judge) |
| **Package Management** | Poetry |
| **Version Control** | Git & GitHub |

---

## 🏃 Quick Start

```bash
# Clone the repo
git clone https://github.com/pallavi-dhadage/genai-learning-journey.git
cd genai-learning-journey

# Install dependencies (using Poetry)
poetry install

# Run the latest agent (Day 8)
poetry run python day-08-langgraph/agent_graph.py
Note: Set up your API keys (e.g., GROQ_API_KEY) in a .env file at the root.

🗓️ Roadmap (Next Steps)
Day 9: Persistent Memory with LangGraph Checkpoints

Day 10: Human‑in‑the‑Loop (breakpoints & approval)

Day 11: Multi‑Tool & Multi‑Agent Orchestration

Day 12: Observability with LangSmith

Days 13–20: Production Optimisation (caching, re‑ranking, cost routing)

Days 21–30: Specialised Capstone (Financial Co‑Pilot)

📝 License
MIT – feel free to use this to kick‑start your own GenAI journey!

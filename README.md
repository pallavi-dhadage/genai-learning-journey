# 🚀 GenAI Learning Journey

A structured, 30‑day, project‑based deep dive into Generative AI Engineering.  
Each day includes a working Python script, a practical concept, and a commit to track real progress.

> **Current Status:** Days 1–12 complete (Foundations → RAG → Agents → Memory → Human Approval → Multi‑Agent Orchestration → Observability).

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
| 08 | LangGraph Agent | State machine agent with tool calling, conversation memory, and dynamic routing. |
| 09 | Persistent Memory (Checkpoints) | LangGraph agent with SQLite checkpoints, supporting multiple conversation threads and memory across restarts. |
| 10 | Human‑in‑the‑Loop | Agent that pauses and asks for human approval before executing sensitive actions (e.g., stock trades). |
| 11 | Multi‑Agent Orchestration | Supervisor pattern with specialised agents: Researcher 📊, Analyst 🧮, and Trader 💰. |
| 12 | **Observability & Multi‑Agent Supervisor** | Integrated LangSmith tracing to visualise Supervisor decisions, agent calls, and tool executions. Built a robust, debugged multi‑agent supervisor (Researcher, Analyst, Trader) with resilient JSON parsing. |

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **LLMs & APIs** | Groq (LLaMA 3.3, LLaMA 3.1), OpenAI (optional) |
| **Embeddings & Retrieval** | Sentence-Transformers, FAISS, scikit-learn |
| **Agent Frameworks** | LangGraph, LangChain Tools |
| **Persistence** | SQLite (checkpoints), Pickle (metadata) |
| **Evaluation** | Manual prompt‑based scoring (Groq as judge) |
| **Human‑in‑the‑Loop** | Direct `input()` approval before tool execution |
| **Orchestration** | Supervisor pattern with specialised agents |
| **Observability** | LangSmith (tracing & debugging) |
| **Resilience** | Robust JSON parsing (`ast.literal_eval` fallback) |
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

# Run the latest multi‑agent supervisor (Day 12)
poetry run python day-12-observability/supervisor_final.py
Note: Set up your API keys (e.g., GROQ_API_KEY) in a .env file at the root.

🗓️ Roadmap (Next Steps)
Day 13: Production Optimisation (Semantic Caching, Cost‑aware Routing)

Day 14: Advanced RAG (HyDE, Re‑ranking)

Day 15: Fine‑tuning with LoRA

Day 16–20: Production Deployment (Docker, FastAPI, Streaming)

Day 21–30: Capstone: Financial Co‑Pilot

📝 License
MIT – feel free to use this to kick‑start your own GenAI journey!
EOF

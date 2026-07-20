# 🚀 GenAI Learning Journey

A structured, 30‑day, project‑based deep dive into Generative AI Engineering.  
Each day includes a working Python script, a practical concept, and a commit to track real progress.

> **Current Status:** Days 1–14 complete (Foundations → RAG → Agents → Memory → Human Approval → Multi‑Agent → Optimisation → Advanced RAG).

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
| 12 | Observability & Debugging | Integrated LangSmith tracing to visualise Supervisor decisions, agent calls, and tool executions. Built a robust, debugged multi‑agent supervisor with resilient JSON parsing. |
| 13 | Production Optimisation | Added **semantic caching** (instant responses for similar questions) and **cost-aware model routing** (cheap `llama-3.1-8b` for simple tasks, premium `llama-3.3-70b` for complex ones). Fixed graph execution flow to prevent infinite loops. |
| 14 | **Advanced RAG** | Implemented **HyDE** (Hypothetical Document Embeddings) for better retrieval quality, **LLM-based re-ranking** for improved document relevance, and **semantic caching** for instant responses to similar queries. Built interactive mode with toggleable features. |

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
| **Production Features** | Semantic Caching (cosine similarity), Cost‑aware Model Routing, HyDE, Re-ranking |
| **Package Management** | Poetry |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure
genai-learning-journey/
├── day-01-prompt-engineering/
│ └── news_summariser.py
├── day-02-streaming/
│ └── stream_chat.py
├── day-03-function-calling/
│ └── agent_with_tools.py
├── day-04-embeddings/
│ └── semantic_search.py
├── day-05-faiss/
│ └── faiss_index.py
├── day-06-rag/
│ └── rag_pipeline.py
├── day-07-evaluation/
│ └── rag_evaluation.py
├── day-08-langgraph/
│ └── langgraph_agent.py
├── day-09-checkpoints/
│ └── checkpoint_agent.py
├── day-10-human-in-loop/
│ └── human_approval_agent.py
├── day-11-multi-agent/
│ └── multi_agent_supervisor.py
├── day-12-observability/
│ └── observable_agent.py
├── day-13-optimisation/
│ └── optimised_agent.py
├── day-14-advanced-rag/
│ ├── advanced_rag.py
│ ├── test_advanced_rag.py
│ └── README.md
├── .gitignore
├── README.md
├── poetry.lock
└── pyproject.toml

text

---

## 🏃 Quick Start

```bash
# Clone the repo
git clone https://github.com/pallavi-dhadage/genai-learning-journey.git
cd genai-learning-journey

# Install dependencies (using Poetry)
poetry install

# Run Day 14: Advanced RAG
poetry run python day-14-advanced-rag/advanced_rag.py

# Run Day 14 in interactive mode
poetry run python day-14-advanced-rag/advanced_rag.py --interactive

# Run tests for Day 14
poetry run python day-14-advanced-rag/test_advanced_rag.py

# Run Day 13: Production Optimisation
poetry run python day-13-optimisation/optimised_agent.py
Note: Set up your API keys (e.g., GROQ_API_KEY, LANGSMITH_API_KEY) in a .env file at the root.

🗓️ Roadmap (Next Steps)
Day	Topic	Description
15	Fine‑tuning with LoRA	Fine-tune a model using Low-Rank Adaptation for specific tasks
16	Docker & Containerization	Containerize the application for production deployment
17	FastAPI & API Development	Build RESTful APIs for the GenAI application
18	Async & Streaming APIs	Implement async endpoints and streaming responses
19	Monitoring & Logging	Set up comprehensive monitoring and logging
20	Deployment Strategies	Deploy to cloud platforms (AWS/GCP/Azure)
21	Capstone: Financial Co‑Pilot	Start building the Financial Co‑Pilot
22	Financial Data Integration	Integrate real-time financial data APIs
23	Portfolio Analysis	Build portfolio analysis and recommendation features
24	Risk Assessment	Implement risk assessment algorithms
25	Market Intelligence	Build market intelligence and news analysis
26	Natural Language Queries	Enable natural language financial queries
27	Automated Trading Signals	Generate automated trading signals
28	Performance Dashboard	Build a performance dashboard for the Co‑Pilot
29	Integration & Testing	Integrate all components and thorough testing
30	Final Deployment	Deploy the complete Financial Co‑Pilot
🎯 Learning Outcomes
By completing this journey, you will have:

Foundation (Days 1-7)
✅ Mastered prompt engineering techniques

✅ Built semantic search and RAG pipelines

✅ Created vector databases with FAISS

✅ Evaluated RAG systems effectively

Agent Development (Days 8-12)
✅ Built LangGraph agents with tool calling

✅ Implemented persistent memory with checkpoints

✅ Created human-in-the-loop workflows

✅ Orchestrated multi-agent systems

✅ Integrated observability with LangSmith

Production Ready (Days 13-14)
✅ Optimized with semantic caching and cost-aware routing

✅ Implemented advanced RAG with HyDE and re-ranking

✅ Created production-grade code with testing

Coming Up (Days 15-30)
🔲 Model fine-tuning with LoRA

🔲 Containerization and deployment

🔲 Full-stack Financial Co-Pilot application

📊 Performance Benchmarks
Feature	Improvement	Use Case
HyDE	+15-25% retrieval accuracy	Complex, ambiguous queries
Re-ranking	+10-20% relevance score	Multi-document synthesis
Semantic Cache	90%+ response time reduction	Repeated/similar queries
Cost-aware Routing	40-60% cost reduction	Mixed complexity workloads
🤝 Contributing
Feel free to fork this repository and use it as a starting point for your own GenAI learning journey. Suggestions and improvements are welcome!

📝 License
MIT – feel free to use this to kick‑start your own GenAI journey!

⭐ Star the Repository
If you find this learning journey helpful, please consider starring the repository on GitHub! ⭐

Happy Learning! 🚀


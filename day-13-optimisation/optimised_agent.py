import os
import json
import ast
import sqlite3
import time
import numpy as np
from typing import TypedDict, Annotated, Literal
import operator
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- SEMANTIC CACHE ----------
class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.queries = []
        self.responses = []
        self.embeddings = []

    def get(self, query: str):
        if not self.queries:
            return None, False
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, np.array(self.embeddings))[0]
        max_sim = np.max(similarities)
        if max_sim >= self.threshold:
            idx = np.argmax(similarities)
            print(f"[CACHE HIT] Similarity: {max_sim:.3f}")
            return self.responses[idx], True
        print(f"[CACHE MISS] Max similarity: {max_sim:.3f}")
        return None, False

    def set(self, query: str, response: dict):
        self.queries.append(query)
        self.responses.append(response)
        embedding = self.model.encode([query])[0]
        self.embeddings.append(embedding)
        print(f"[CACHE STORED] Query: {query[:30]}...")

# ---------- COST-AWARE ROUTER ----------
def route_model(messages):
    full_text = " ".join([m.get("content", "") for m in messages if m.get("content")])
    word_count = len(full_text.split())
    premium_keywords = ["trade", "buy", "sell", "risk", "portfolio", "analysis", "detailed", "explain", "compare"]
    if word_count > 50 or any(kw in full_text.lower() for kw in premium_keywords):
        model = "llama-3.3-70b-versatile"
        print(f"[ROUTER] Using Premium model (word_count={word_count})")
    else:
        model = "llama-3.1-8b-instant"
        print(f"[ROUTER] Using Cheap model (word_count={word_count})")
    return model

# ---------- GLOBAL CACHE ----------
cache = SemanticCache(threshold=0.95)

# ---------- OPTIMISED LLM CALL ----------
def call_llm(messages, tools=None, route=True):
    cleaned = []
    for m in messages:
        cleaned_m = {}
        for k in ["role", "content", "tool_calls", "tool_call_id"]:
            if k in m and m[k] is not None:
                cleaned_m[k] = m[k]
        cleaned.append(cleaned_m)

    is_pure_chat = tools is None
    cache_key = None
    if is_pure_chat:
        user_msgs = [m for m in cleaned if m.get("role") == "user"]
        if user_msgs:
            cache_key = user_msgs[-1].get("content", "")

    if cache_key:
        cached_response, found = cache.get(cache_key)
        if found:
            return cached_response

    if route and is_pure_chat:
        model = route_model(cleaned)
    else:
        model = "llama-3.3-70b-versatile"

    params = {
        "model": model,
        "messages": cleaned,
        "temperature": 0.3
    }
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    start = time.time()
    response = client.chat.completions.create(**params)
    latency = time.time() - start
    print(f"[LATENCY] {latency:.2f}s")

    result = response.choices[0].message.model_dump()

    if is_pure_chat and result.get("content") and cache_key:
        cache.set(cache_key, result)

    return result

# ---------- TOOLS ----------
def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Math Error: {e}"

def get_stock_price(symbol: str) -> str:
    mock = {"AAPL": "185.23", "TSLA": "256.40", "MSFT": "420.10"}
    return mock.get(symbol.upper(), "Unknown symbol")

def execute_trade(symbol: str, quantity: int, action: str) -> str:
    return f"✅ Trade executed: {action} {quantity} shares of {symbol} at market price."

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Gets a mock stock price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_trade",
            "description": "Executes a stock trade (BUY or SELL). This requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                    "quantity": {"type": "integer", "description": "Number of shares"},
                    "action": {"type": "string", "description": "BUY or SELL", "enum": ["BUY", "SELL"]}
                },
                "required": ["symbol", "quantity", "action"]
            }
        }
    }
]

# ---------- STATE ----------
class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    next_step: str

def clean_message(msg: dict) -> dict:
    cleaned = {}
    for k in ["role", "content", "tool_calls", "tool_call_id"]:
        if k in msg and msg[k] is not None:
            cleaned[k] = msg[k]
    return cleaned

# ---------- SUPERVISOR ----------
def supervisor(state: AgentState):
    system = """You are a Supervisor. Route to:
- "researcher" for stock prices
- "analyst" for mathematical calculations
- "trader" for buying/selling stocks

Output ONLY a valid JSON object with a single key "next". 
Examples: {"next": "researcher"} or {"next": "FINISH"}"""
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, route=True)
    content = response.get("content", "").strip()
    print(f"[DEBUG] Supervisor raw: {content}")

    next_step = "FINISH"
    try:
        decision = ast.literal_eval(content)
        if isinstance(decision, dict):
            next_step = decision.get("next", "FINISH")
    except:
        try:
            decision = json.loads(content)
            if isinstance(decision, dict):
                next_step = decision.get("next", "FINISH")
        except:
            if content in ["researcher", "analyst", "trader", "FINISH"]:
                next_step = content

    print(f"[DEBUG] Supervisor decided: {next_step}")
    return {"next_step": next_step}

# ---------- AGENTS ----------
def researcher_agent(state: AgentState):
    print("[DEBUG] Researcher agent called")
    system = "You are a Researcher. Use get_stock_price."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[1]], route=True)
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = get_stock_price(**args)
            final_answer = f"The current price of {args['symbol'].upper()} is ${result}."
            return {
                "messages": [{"role": "assistant", "content": final_answer}],
                "next_step": "FINISH"
            }
    content = response.get("content", "")
    if not content:
        content = "I couldn't find that stock price."
    return {
        "messages": [{"role": "assistant", "content": content}],
        "next_step": "FINISH"
    }

def analyst_agent(state: AgentState):
    print("[DEBUG] Analyst agent called")
    system = "You are an Analyst. Use calculate."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[0]], route=True)
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = calculate(**args)
            final_answer = f"The result is {result}."
            return {
                "messages": [{"role": "assistant", "content": final_answer}],
                "next_step": "FINISH"
            }
    content = response.get("content", "")
    if not content:
        content = "I couldn't perform that calculation."
    return {
        "messages": [{"role": "assistant", "content": content}],
        "next_step": "FINISH"
    }

def trader_agent(state: AgentState):
    print("[DEBUG] Trader agent called")
    system = "You are a Trader. Use execute_trade."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[2]], route=True)
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            print("\n⚠️  TRADE REQUESTED ⚠️")
            print(f"Symbol: {args.get('symbol')}, Qty: {args.get('quantity')}, Action: {args.get('action')}")
            approval = input("Approve? (yes/no): ")
            if approval.lower() == "yes":
                final_answer = execute_trade(**args)
            else:
                final_answer = "❌ Trade rejected by user."
            return {
                "messages": [{"role": "assistant", "content": final_answer}],
                "next_step": "FINISH"
            }
    content = response.get("content", "")
    if not content:
        content = "I couldn't process that trade."
    return {
        "messages": [{"role": "assistant", "content": content}],
        "next_step": "FINISH"
    }

# ---------- ROUTER ----------
def route_after_supervisor(state):
    next_step = state.get("next_step", "FINISH")
    if next_step == "FINISH":
        return END
    return next_step

# ---------- BUILD GRAPH ----------
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("trader", trader_agent)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_after_supervisor, {
    "researcher": "researcher",
    "analyst": "analyst",
    "trader": "trader",
    END: END
})

# FIX: Agents go to END directly (not back to supervisor)
workflow.add_edge("researcher", END)
workflow.add_edge("analyst", END)
workflow.add_edge("trader", END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

# ---------- RUN ----------
print("🤖 Optimised Supervisor Agent (Caching + Routing)")
print("💡 First time: Slow (LLM call). Second time: Instant (Cache).")
print("📊 Router: Simple greetings use cheap model, complex trades use premium.\n")

state = {"messages": []}
while True:
    cmd = input("You: ")
    if cmd.lower() == "exit":
        break
    state["messages"].append({"role": "user", "content": cmd})
    final = graph.invoke(state, config={"configurable": {"thread_id": "default"}})
    msgs = [m for m in final["messages"] if m.get("role") == "assistant" and m.get("content")]
    if msgs:
        print(f"Assistant: {msgs[-1]['content']}\n")
    state = final

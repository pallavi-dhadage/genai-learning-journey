import os
import json
import sqlite3
from typing import TypedDict, Annotated, Literal
import operator
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Tools (Same as Day 11) ----------
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

tool_map = {
    "calculate": calculate,
    "get_stock_price": get_stock_price,
    "execute_trade": execute_trade
}

tool_schemas = [
    {"type": "function", "function": {"name": "calculate", "description": "Evaluates a mathematical expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "description": "Gets a mock stock price.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "execute_trade", "description": "Executes a stock trade. Requires approval.", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "quantity": {"type": "integer"}, "action": {"type": "string", "enum": ["BUY", "SELL"]}}, "required": ["symbol", "quantity", "action"]}}}
]

# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    next_step: str

def clean_message(msg: dict) -> dict:
    cleaned = {}
    for k in ["role", "content", "tool_calls", "tool_call_id"]:
        if k in msg and msg[k] is not None:
            cleaned[k] = msg[k]
    return cleaned

def call_llm(messages, tools=None):
    cleaned = [clean_message(m) for m in messages]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=cleaned,
        tools=tools,
        tool_choice="auto" if tools else None,
        temperature=0.3
    )
    return response.choices[0].message.model_dump()

# ---------- Nodes ----------
def supervisor(state: AgentState):
    system = "You are a Supervisor. Route to 'researcher' for stocks, 'analyst' for math, 'trader' for trades. Output JSON: {'next': '...'} or 'FINISH'."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs)
    try:
        decision = json.loads(response["content"])
        next_step = decision.get("next", "FINISH")
    except:
        next_step = "FINISH"
    return {"next_step": next_step}

def researcher_agent(state: AgentState):
    system = "You are a Researcher. Use get_stock_price."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[1]])
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = get_stock_price(**args)
            return {"messages": [{"role": "assistant", "content": response.get("content", "")}, {"role": "tool", "tool_call_id": tc["id"], "content": result}], "next_step": "supervisor"}
    return {"messages": [{"role": "assistant", "content": response.get("content", "")}], "next_step": "supervisor"}

def analyst_agent(state: AgentState):
    system = "You are an Analyst. Use calculate."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[0]])
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = calculate(**args)
            return {"messages": [{"role": "assistant", "content": response.get("content", "")}, {"role": "tool", "tool_call_id": tc["id"], "content": result}], "next_step": "supervisor"}
    return {"messages": [{"role": "assistant", "content": response.get("content", "")}], "next_step": "supervisor"}

def trader_agent(state: AgentState):
    system = "You are a Trader. Use execute_trade."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[2]])
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            print("\n⚠️  TRADE REQUESTED ⚠️")
            print(f"Symbol: {args.get('symbol')}, Qty: {args.get('quantity')}, Action: {args.get('action')}")
            approval = input("Approve? (yes/no): ")
            result = execute_trade(**args) if approval.lower() == "yes" else "❌ Rejected."
            return {"messages": [{"role": "assistant", "content": response.get("content", "")}, {"role": "tool", "tool_call_id": tc["id"], "content": result}], "next_step": "supervisor"}
    return {"messages": [{"role": "assistant", "content": response.get("content", "")}], "next_step": "supervisor"}

def route_after_supervisor(state):
    return state.get("next_step", "FINISH")

# ---------- Build Graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("trader", trader_agent)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_after_supervisor, {"researcher": "researcher", "analyst": "analyst", "trader": "trader", END: END})
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("trader", "supervisor")

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
graph = workflow.compile(checkpointer=SqliteSaver(conn))

# ---------- Run ----------
print("🤖 Supervisor Agent (Working version) - type 'exit' to quit\n")
state = {"messages": []}
while True:
    cmd = input("You: ")
    if cmd.lower() == "exit": break
    state["messages"].append({"role": "user", "content": cmd})
    final = graph.invoke(state, config={"configurable": {"thread_id": "default"}})
    msgs = [m for m in final["messages"] if m.get("role") == "assistant" and m.get("content")]
    if msgs: print(f"Assistant: {msgs[-1]['content']}")
    state = final

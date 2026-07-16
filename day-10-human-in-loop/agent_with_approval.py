import os
import json
import sqlite3
from typing import TypedDict, Annotated
import operator
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Tools ----------
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

# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]

def clean_message(msg: dict) -> dict:
    cleaned = {}
    if "role" in msg:
        cleaned["role"] = msg["role"]
    if "content" in msg and msg["content"] is not None:
        cleaned["content"] = msg["content"]
    if "tool_calls" in msg and msg["tool_calls"] is not None:
        cleaned["tool_calls"] = msg["tool_calls"]
    if "tool_call_id" in msg:
        cleaned["tool_call_id"] = msg["tool_call_id"]
    return cleaned

# ---------- Nodes ----------
def call_model(state: AgentState):
    messages = state["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a helpful financial assistant."}] + messages
    clean_msgs = [clean_message(m) for m in messages]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        tools=tool_schemas,
        tool_choice="auto",
        temperature=0.3
    )
    raw_msg = response.choices[0].message.model_dump()
    assistant_msg = clean_message(raw_msg)
    return {"messages": [assistant_msg]}

def call_tool(state: AgentState):
    last = state["messages"][-1]
    tool_calls = last.get("tool_calls", [])
    results = []
    for tc in tool_calls:
        func_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        
        # ---------- HUMAN-IN-THE-LOOP (SIMPLE PROMPT) ----------
        if func_name == "execute_trade":
            print("\n⚠️  SENSITIVE ACTION DETECTED ⚠️")
            print(f"🔹 Symbol: {args.get('symbol')}")
            print(f"🔹 Quantity: {args.get('quantity')}")
            print(f"🔹 Action: {args.get('action')}")
            
            # We ask for approval directly here using input().
            # The graph PAUSES right here until we type.
            approval = input("🤔 Approve this trade? (yes/no): ")
            
            if approval.lower() == "yes":
                result = execute_trade(**args)
            else:
                result = "❌ Trade rejected by user."
        else:
            func = tool_map.get(func_name)
            if func:
                result = func(**args)
            else:
                result = f"Unknown tool: {func_name}"
        
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(result)
        })
    return {"messages": results}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.get("tool_calls"):
        return "tools"
    return "end"

# ---------- Build Graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

# ---------- Interactive Loop ----------
print("🤖 LangGraph Agent with Human‑in‑the‑Loop (type 'exit' to quit)")
print("💡 The agent will pause and ask for your approval before executing trades.\n")

thread_id = "default_user"
config = {"configurable": {"thread_id": thread_id}}
state = {"messages": []}

while True:
    cmd = input("You: ")
    if cmd.lower() == "exit":
        break
    
    state["messages"].append({"role": "user", "content": cmd})
    final_state = graph.invoke(state, config=config)
    
    assistant_msgs = [m for m in final_state["messages"] if m.get("role") == "assistant"]
    if assistant_msgs and assistant_msgs[-1].get("content"):
        print(f"Assistant: {assistant_msgs[-1]['content']}")
    else:
        print("Assistant: (processing completed)")
    
    state = final_state
    print(f"📝 Total messages in thread: {len(state['messages'])}\n")

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

tool_map = {
    "calculate": calculate,
    "get_stock_price": get_stock_price
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
    }
]

# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]

# ---------- Helper to clean messages ----------
def clean_message(msg: dict) -> dict:
    """Return a copy with only fields Groq accepts, and omit tool_calls if None."""
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
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
    
    # Clean each message – remove fields Groq doesn't like
    clean_msgs = [clean_message(m) for m in messages]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        tools=tool_schemas,
        tool_choice="auto",
        temperature=0.3
    )
    raw_msg = response.choices[0].message.model_dump()
    # Clean the assistant message before returning
    assistant_msg = clean_message(raw_msg)
    return {"messages": [assistant_msg]}

def call_tool(state: AgentState):
    last = state["messages"][-1]
    tool_calls = last.get("tool_calls", [])
    results = []
    for tc in tool_calls:
        func_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        func = tool_map.get(func_name)
        if func:
            result = func(**args)
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

# ---------- Build Graph with Checkpointer ----------
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
print("🤖 LangGraph Agent with Persistent Memory (type 'exit' to quit)")
print("💡 Tip: Use 'switch' to change thread, e.g., 'switch work_chat'\n")

thread_id = "default_user"
config = {"configurable": {"thread_id": thread_id}}

while True:
    print(f"Current Thread: {thread_id}")
    cmd = input("You: ")
    
    if cmd.lower() == "exit":
        break
    elif cmd.lower().startswith("switch"):
        parts = cmd.split()
        if len(parts) > 1:
            new_thread = parts[1]
            thread_id = new_thread
            config = {"configurable": {"thread_id": thread_id}}
            print(f"✅ Switched to thread: {thread_id}\n")
        else:
            print("Usage: switch <thread_id>\n")
        continue
    
    state = {"messages": [{"role": "user", "content": cmd}]}
    final_state = graph.invoke(state, config=config)
    
    assistant_msgs = [m for m in final_state["messages"] if m.get("role") == "assistant"]
    if assistant_msgs:
        print(f"Assistant: {assistant_msgs[-1]['content']}")
    
    print(f"📝 Total messages in this thread: {len(final_state['messages'])}\n")


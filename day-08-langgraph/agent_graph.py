import os
import json
from typing import TypedDict, Annotated
import operator
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

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

# ---------- Nodes ----------
def call_model(state: AgentState):
    messages = state["messages"]
    # Ensure system message
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
    # Clean: only role, content, tool_calls (Groq doesn't want extra fields)
    clean = []
    for m in messages:
        allowed = {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}}
        # Also ensure tool_call_id is only present for tool messages
        if m.get("role") == "tool" and "tool_call_id" not in allowed:
            # Should not happen, but let's be safe
            continue
        clean.append(allowed)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Use active model
        messages=clean,
        tools=tool_schemas,
        tool_choice="auto",
        temperature=0.3
    )
    msg = response.choices[0].message.model_dump()
    # Keep only allowed fields
    msg = {k: v for k, v in msg.items() if k in {"role", "content", "tool_calls"}}
    return {"messages": [msg]}

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
                "tool_call_id": tc["id"],   # MUST include this!
                "content": str(result)
            })
    return {"messages": results}

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if last.get("tool_calls"):
        return "tools"
    return "end"

# ---------- Graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# ---------- Interactive Loop ----------
print("🤖 LangGraph Agent (type 'exit' to quit)\n")
state = {"messages": []}

while True:
    user = input("You: ")
    if user.lower() in ["exit", "quit"]:
        break
    state["messages"].append({"role": "user", "content": user})
    final_state = graph.invoke(state)
    # Extract assistant messages
    assistant_msgs = [m for m in final_state["messages"] if m.get("role") == "assistant"]
    if assistant_msgs:
        print(f"Assistant: {assistant_msgs[-1]['content']}")
    state = final_state

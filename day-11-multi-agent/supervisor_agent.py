import os
import json
import sqlite3
import time
from typing import TypedDict, Annotated, Literal
import operator
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Retry Decorator ----------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def groq_chat_completion(**kwargs):
    return client.chat.completions.create(**kwargs)

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

researcher_tools = {"get_stock_price": get_stock_price}
analyst_tools = {"calculate": calculate}
trader_tools = {"execute_trade": execute_trade}

tool_schemas = {
    "researcher": [
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
    ],
    "analyst": [
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
        }
    ],
    "trader": [
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
}

class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    next_step: str

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

def supervisor(state: AgentState):
    messages = state["messages"]
    system = """You are a Supervisor. Your job is to route the user's request to the right specialist agent.

Available agents:
- researcher: Handles questions about stock prices (e.g., 'What is the price of Tesla?').
- analyst: Handles mathematical calculations (e.g., 'What is 25 * 4?').
- trader: Handles stock trades (e.g., 'Buy 100 shares of Apple').

If the user is just greeting or saying thanks, respond directly without routing to an agent.

Output your decision as a JSON object with a single key "next" and one of the values: "researcher", "analyst", "trader", or "FINISH".
Examples:
{"next": "researcher"}
{"next": "FINISH"}
"""
    if not messages or messages[0].get("role") != "system":
        full_msgs = [{"role": "system", "content": system}] + messages
    else:
        full_msgs = messages
    clean_msgs = [clean_message(m) for m in full_msgs]
    response = groq_chat_completion(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    try:
        decision = json.loads(content)
        next_step = decision.get("next", "FINISH")
    except:
        next_step = "FINISH"
    return {"next_step": next_step}

def researcher_agent(state: AgentState):
    messages = state["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a Researcher. Fetch stock prices using get_stock_price."}] + messages
    clean_msgs = [clean_message(m) for m in messages]
    response = groq_chat_completion(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        tools=tool_schemas["researcher"],
        tool_choice="auto",
        temperature=0.3
    )
    raw_msg = response.choices[0].message.model_dump()
    assistant_msg = clean_message(raw_msg)
    return {"messages": [assistant_msg], "next_step": "supervisor"}

def analyst_agent(state: AgentState):
    messages = state["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are an Analyst. Perform calculations using calculate."}] + messages
    clean_msgs = [clean_message(m) for m in messages]
    response = groq_chat_completion(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        tools=tool_schemas["analyst"],
        tool_choice="auto",
        temperature=0.3
    )
    raw_msg = response.choices[0].message.model_dump()
    assistant_msg = clean_message(raw_msg)
    return {"messages": [assistant_msg], "next_step": "supervisor"}

def trader_agent(state: AgentState):
    messages = state["messages"]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": "You are a Trader. Execute trades using execute_trade. Always ask for approval."}] + messages
    clean_msgs = [clean_message(m) for m in messages]
    response = groq_chat_completion(
        model="llama-3.3-70b-versatile",
        messages=clean_msgs,
        tools=tool_schemas["trader"],
        tool_choice="auto",
        temperature=0.3
    )
    raw_msg = response.choices[0].message.model_dump()
    assistant_msg = clean_message(raw_msg)
    if assistant_msg.get("tool_calls"):
        results = []
        for tc in assistant_msg["tool_calls"]:
            func_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            if func_name == "execute_trade":
                print("\n⚠️  TRADE REQUESTED ⚠️")
                print(f"🔹 Symbol: {args.get('symbol')}")
                print(f"🔹 Quantity: {args.get('quantity')}")
                print(f"🔹 Action: {args.get('action')}")
                approval = input("🤔 Approve this trade? (yes/no): ")
                if approval.lower() == "yes":
                    result = execute_trade(**args)
                else:
                    result = "❌ Trade rejected by user."
                results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result)
                })
        return {"messages": results, "next_step": "supervisor"}
    return {"messages": [assistant_msg], "next_step": "supervisor"}

def execute_agent_tools(state: AgentState):
    last = state["messages"][-1]
    tool_calls = last.get("tool_calls", [])
    if not tool_calls:
        return {"messages": []}
    results = []
    all_tools = {
        "get_stock_price": get_stock_price,
        "calculate": calculate
    }
    for tc in tool_calls:
        func_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        func = all_tools.get(func_name)
        if func:
            result = func(**args)
            results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)
            })
    return {"messages": results}

def route_after_supervisor(state: AgentState) -> Literal["researcher", "analyst", "trader", END]:
    next_step = state.get("next_step", "FINISH")
    if next_step == "FINISH":
        return END
    return next_step

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("trader", trader_agent)
workflow.add_node("execute_tools", execute_agent_tools)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_after_supervisor, {
    "researcher": "researcher",
    "analyst": "analyst",
    "trader": "trader",
    END: END
})
workflow.add_edge("researcher", "execute_tools")
workflow.add_edge("execute_tools", "supervisor")
workflow.add_edge("analyst", "execute_tools")
workflow.add_edge("trader", "supervisor")

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

print("🤖 Multi‑Agent Supervisor (type 'exit' to quit)")
print("Agents: Researcher 📊, Analyst 🧮, Trader 💰\n")

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

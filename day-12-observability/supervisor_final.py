import os
import json
import ast
import sqlite3
from typing import TypedDict, Annotated, Literal
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
    next_step: str

def clean_message(msg: dict) -> dict:
    cleaned = {}
    for k in ["role", "content", "tool_calls", "tool_call_id"]:
        if k in msg and msg[k] is not None:
            cleaned[k] = msg[k]
    return cleaned

def call_llm(messages, tools=None):
    cleaned = [clean_message(m) for m in messages]
    params = {
        "model": "llama-3.1-8b-instant",
        "messages": cleaned,
        "temperature": 0.1  # Lower temperature for deterministic routing
    }
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    response = client.chat.completions.create(**params)
    return response.choices[0].message.model_dump()

# ---------- Nodes ----------
def supervisor(state: AgentState):
    system = """You are a Supervisor. Route to:
- "researcher" for stock prices
- "analyst" for mathematical calculations
- "trader" for buying/selling stocks

If you cannot route, use "FINISH".

Output ONLY a valid JSON object with a single key "next". 
Examples:
{"next": "researcher"}
{"next": "FINISH"}
Do NOT output anything else. No explanations, no text, only the JSON object."""

    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs)
    content = response.get("content", "").strip()
    print(f"[DEBUG] Supervisor raw: {content}")

    next_step = "FINISH"
    try:
        # Try parsing with ast.literal_eval first (handles single quotes)
        decision = ast.literal_eval(content)
        if isinstance(decision, dict):
            next_step = decision.get("next", "FINISH")
        else:
            # If it's not a dict, check if it's a string like 'FINISH'
            if isinstance(decision, str):
                if decision in ["researcher", "analyst", "trader", "FINISH"]:
                    next_step = decision
    except:
        try:
            # Fallback to JSON parsing
            decision = json.loads(content)
            if isinstance(decision, dict):
                next_step = decision.get("next", "FINISH")
        except:
            # Last resort: check if content itself is a valid keyword
            if content in ["researcher", "analyst", "trader", "FINISH"]:
                next_step = content

    print(f"[DEBUG] Supervisor decided: {next_step}")
    return {"next_step": next_step}

def researcher_agent(state: AgentState):
    print("[DEBUG] Researcher agent called")
    system = "You are a Researcher. Use get_stock_price."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[1]])

    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = get_stock_price(**args)
            final_answer = f"The current price of {args['symbol'].upper()} is ${result}."
            return {
                "messages": [{"role": "assistant", "content": final_answer}],
                "next_step": "supervisor"
            }

    # If no tool call, just return the assistant's response
    content = response.get("content", "")
    if not content:
        content = "I couldn't find that stock price."
    return {"messages": [{"role": "assistant", "content": content}], "next_step": "supervisor"}

def analyst_agent(state: AgentState):
    print("[DEBUG] Analyst agent called")
    system = "You are an Analyst. Use calculate."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[0]])

    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])
            result = calculate(**args)
            final_answer = f"The result is {result}."
            return {
                "messages": [{"role": "assistant", "content": final_answer}],
                "next_step": "supervisor"
            }

    content = response.get("content", "")
    if not content:
        content = "I couldn't perform that calculation."
    return {"messages": [{"role": "assistant", "content": content}], "next_step": "supervisor"}

def trader_agent(state: AgentState):
    print("[DEBUG] Trader agent called")
    system = "You are a Trader. Use execute_trade."
    msgs = [{"role": "system", "content": system}] + state["messages"]
    response = call_llm(msgs, tools=[tool_schemas[2]])

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
                "next_step": "supervisor"
            }

    content = response.get("content", "")
    if not content:
        content = "I couldn't process that trade."
    return {"messages": [{"role": "assistant", "content": content}], "next_step": "supervisor"}

def route_after_supervisor(state):
    next_step = state.get("next_step", "FINISH")
    if next_step == "FINISH":
        return END
    return next_step

# ---------- Build Graph ----------
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
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("trader", "supervisor")

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

# ---------- Run ----------
print("🤖 Supervisor Agent (Fixed) - type 'exit' to quit\n")
state = {"messages": []}
while True:
    cmd = input("You: ")
    if cmd.lower() == "exit":
        break
    state["messages"].append({"role": "user", "content": cmd})
    final = graph.invoke(state, config={"configurable": {"thread_id": "default"}})
    msgs = [m for m in final["messages"] if m.get("role") == "assistant" and m.get("content")]
    if msgs:
        print(f"Assistant: {msgs[-1]['content']}")
    else:
        print("[DEBUG] No assistant content in final state.")
    state = final

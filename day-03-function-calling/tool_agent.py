import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =============================================
# 1. Define the actual Python tools (functions)
# =============================================

def get_stock_price(symbol: str) -> str:
    """Mock function to simulate fetching a stock price."""
    # In reality, you'd call yfinance or an API here.
    mock_prices = {
        "AAPL": "185.23 USD (+2.1%)",
        "TSLA": "256.40 USD (-1.3%)",
        "MSFT": "420.10 USD (+0.8%)",
        "GOOGL": "175.50 USD (+0.5%)"
    }
    return mock_prices.get(symbol.upper(), f"Stock {symbol.upper()} not found.")

def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    # We restrict builtins for security – only allow math operations.
    allowed = {"abs": abs, "round": round}
    try:
        # Use eval with restricted globals to prevent code injection.
        return eval(expression, {"__builtins__": allowed}, {})
    except Exception as e:
        return f"Math Error: {e}"

# =============================================
# 2. Define the tool schemas (what the model sees)
# =============================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current mock price for a given stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, TSLA)."
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression (e.g., (21.6 - 20.1) / 20.1 * 100)."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# =============================================
# 3. Map tool names to actual Python functions
# =============================================

available_functions = {
    "get_stock_price": get_stock_price,
    "calculate": calculate
}

# =============================================
# 4. Main agent loop
# =============================================

messages = [
    {"role": "system", "content": "You are a financial assistant. Use the provided tools to answer questions about stock prices and perform calculations."}
]

print("🤖 QuantAgent with Tools (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append user message to conversation history
    messages.append({"role": "user", "content": user_input})

    # Step 1: Ask the model if it wants to use a tool
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let the model decide
        temperature=0.3
    )

    # Capture the model's response message
    response_message = response.choices[0].message
    messages.append(response_message)  # Add it to history

    # Step 2: Check if the model wants to call a tool
    tool_calls = response_message.tool_calls

    if tool_calls:
        # We'll process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Get the Python function and call it
            func = available_functions.get(function_name)
            if func:
                result = func(**function_args)
                # Store the tool result as a new message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Tool {function_name} not found."
                })

        # Step 3: Ask the model to generate the final answer using the tool results
        second_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3
        )
        final_answer = second_response.choices[0].message.content
        print(f"Assistant: {final_answer}\n")
        messages.append({"role": "assistant", "content": final_answer})

    else:
        # No tool call needed – just print the model's direct answer
        print(f"Assistant: {response_message.content}\n")
        messages.append({"role": "assistant", "content": response_message.content})

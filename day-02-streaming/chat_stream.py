import os
from groq import Groq
from dotenv import load_dotenv

# 1. Load environment variables (GROQ_API_KEY)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Initialise the conversation history
messages = [
    {"role": "system", "content": "You are a helpful financial assistant. Answer concisely and accurately."}
]

print("🤖 Welcome to QuantChat! (type 'exit' to quit)\n")

# 3. Main chat loop
while True:
    # 3a. Get user input
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    # 3b. Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # 3c. Make the streaming API call
    print("Assistant: ", end="", flush=True)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        stream=True  # <-- THIS is the magic line
    )
    
    # 3d. Process the streamed chunks
    full_response = ""
    for chunk in response:
        # Extract the content delta from the chunk
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta
    
    print("\n")  # add a newline after the stream ends
    
    # 3e. Add assistant's full response to history
    messages.append({"role": "assistant", "content": full_response})

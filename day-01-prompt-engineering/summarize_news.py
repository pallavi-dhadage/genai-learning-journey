import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_article(text: str, max_length: int = 50) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Most capable, recommended replacement  
        messages=[
            {"role": "system", "content": f"You are a financial news summariser. Summarise the following article in under {max_length} words."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

sample = """
Tesla reported record quarterly deliveries of 484,507 vehicles, beating estimates.
The company also announced a new battery factory in Nevada and a price cut in China.
Shares rose 5% in after-hours trading.
"""
print(summarize_article(sample))

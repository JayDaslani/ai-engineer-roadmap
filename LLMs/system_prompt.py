
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful coding teacher. Explain answers in simple Hinglish with examples."
        },
        {
            "role": "user",
            "content": "What is the difference between system prompt and user prompt?"
        }
    ]
)

print(response.choices[0].message.content)
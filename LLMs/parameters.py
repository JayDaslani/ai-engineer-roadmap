from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

question = "Write essay for diwali"

print("=== Temperature 0.0 ===")
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages = [{
        'role': 'user',
        'content': question
    }],
    temperature=0.0,
    max_tokens=100
)
print(response.choices[0].message.content)

print("=== Temperature 1.0 ===")
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages = [{
        'role': 'user',
        'content': question
    }],
    temperature=1.0,
    max_tokens=100
)
print(response.choices[0].message.content)

print("=== Max token 20 ===")
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages = [{
        'role': 'user',
        'content': question
    }],
    temperature=0.7,
    max_tokens=100
)
print(response.choices[0].message.content)






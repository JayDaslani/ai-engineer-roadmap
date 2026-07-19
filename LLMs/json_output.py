from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def ask(prompt):
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages = [
            {
                'role': 'system',
                'content': """You are a data extraction assistant.
                Always return valid JSON. No extra text. Only JSON."""
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature = 0.1,
        max_tokens = 300
    )
    return response.choices[0].message.content

print("=== Test 1: Data Extraction ===")
result = ask("""
Extract information from this text in JSON format: 
    "Jay Dasalani is a 21 years old AI/ML student who live in ahmedbad and knows python." 
    JSON format :
        {
             "name" : "",
             "age": 0,
             "field": "",
             "city": "",
             "skill": ""
        }""")

print(result)

try:
    data = json.loads(result)
    print('Parsed successfully')
    print(f"Name : {data['name']}")
    print(f"City : {data['city']}")
except Exception as e:
    print(f"JSON is not parsed {e}")

print("=== Test 2 : List Format ===")
result = ask("""
Tell me the top 3 uses of Python in JSON format :
{
    "uses": [
        {"number": 1, "use": "", "example": ""},
        {"number": 2, "use": "", "example": ""}
        {"number": 3, "use": "", "example": ""}
    ]
}""")
print(result)

try:
    data = json.loads(result)
    print("Pyhton uses : ")
    for item in data['uses']:
        print(f"{item['number']}. {item['use']}")
except Exception as e:
    print(f"parse failed: {e}")

print("=== Test 3: Sentiment JSON ===")
reviews = [
    'The product is very good!',
    "The delivery was slow",
    "The price is a bit high"
]

for review in reviews:
    result = ask(f"""
Analyze the following reviews in JSON format:
        Reviews : "{review}"

{{
   "review": "",
   "sentiment": "",
   "score": 0,
   "reason": ""
}}""")
    
print(f"Review : {review}")
print(f"Analysis : {result}")

try:
    data = json.loads(result)
    print(f"Parsed Sentiment: {data['sentiment']}\n")
except:
    print("Test 3 parse failed\n")

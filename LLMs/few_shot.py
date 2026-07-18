from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def ask(prompt):
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages = [
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )
    return response.choices[0].message.content

print("=== ZERO SHOT ===")
zero_shot = """
Tell me the sentiment : 
Text: Today's meeting was very boring"""
print(ask(zero_shot))

print("=== FEW SHOT ===")
few_shot = """
Here are some examples to understand the pattern. Tell me the sentiment for the following sentence:
Text: Today was a very nice day.
Sentiment: Positive
Text: The food was very bad.
Sentiment: Negative
Text: The movie was awesome.
Sentiment: Positive
Text: I failed today's job interview
Sentiment: Negative
Text: Today's meeting was very boring.
Sentiment: """
print(ask(few_shot))

print("=== FEW SHOT Translation ===")
translation = """
Translate the sentece hindi to English.
follow the same format:

Hindi: Mera naam Jay hai.
English: My name is Jay.

Hindi: Aaj mausam accha hai.
English: Today the weather is nice.

Hindi: Main pune mein rehta hoon.
English: I live in pune.

Translate this:
Hindi: Main ek ai engineer banna chahta hoon.
English: 
"""
print(ask(translation))

print("=== FEW SHOT Data Extraction ===")
extraction = """
Find out name and text in Text.
Follow the same format:

Text: Rahul live in Mumbai.
Name: Rahul
City: Mumbai

Text: Priya is from delhi.
Name: Priya
City: delhi

let's find.
Text: Jay is studying pune.
Name:
City:
 """

print(ask(extraction))
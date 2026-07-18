from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def ask(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages = [
            {'role': "user", 'content': prompt}
        ],
        temperature=temperature,
        max_tokens=300
    )

    return response.choices[0].message.content

print("===part 1: Vegue Prompt====")
print(ask('Tell me about python '))


print("===Test 1: clear prompt===")
print(ask("""Tell me about python:
- describe only in 3 points
- In simple english
- for begineer"""))

print("===Test 2: with context===")
print(ask("""
This python code is trying to print 'hello world', but it is trowing a error
          print(Hello)
          Error: NameError: name 'Hello' is not defined
Please fix the code and explain what went wrong."""))

print('===Test 3: Format specified')
print(ask("""
give the data science top 5 skills
          Format:
          1. skill name -1 line explanation
          2. skill name -2 line explanation
          ... as it is
          give only list - not give extra text"""))


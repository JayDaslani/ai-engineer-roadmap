from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def ask(prompt, temperature=0.3):
    response = client.chat.completions.create(
        model = 'llama-3.3-70b-versatile',
        messages = [
            {'role': 'user', 'content': prompt}
        ],
        temperature = temperature,
        max_tokens = 500
    )

    return response.choices[0].message.content

print('=== without cot ===')
print(ask('If a train travels at a speed of 60 km/h, how much distance will it cover in 2.5 hours?'))

print('With Cot')
print(ask("""
If a train travels at a speed of 60 km/h, how much distance will it cover in 2.5 hours? solve it
    step by step:
    step 1 : write the formula
    step 2 : substitute the values
    step 3 : calculate
    step 4 : give me the answer"""))

print('Logic without cot')
print(ask('Jay has 5 apples. He gave 2 apples to raj. Then Mom gave him 3 more. How many apples does Jay have now?'))

print('Logic with cot')
print(ask("""
Jay has 5 apples. He gave 2 apples to raj. Then Mom gave him 3 more. How many apples does Jay have now?
solve it step by step :"""))

print('Decision with cot')
print(ask("""
I need to decide whether to learn data science or web development. about me: 
    I am an AI/ML branch student.
    I know python
    I want to start a business in the future.
    I want a job in pune.
    
    Analyze it step-by-step:
Step 1: Compare both fields
Step 2: Evaluate my situation
Step 3: Provide a recommendation with reasons"""))


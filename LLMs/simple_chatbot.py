from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))



def chat():

    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant'
    }]

    print("===chatbot===")
    print('Press quit for close')

    while True:

        user_input = input('You : ')

        if user_input.lower() == 'quit':
            print('Bye')
            break

        messages.append({
            'role': 'user',
            'content': user_input
        })

        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=messages,
            temperature=0.2,
            max_tokens=200
        )

        ai_response = response.choices[0].message.content
        print(f"AI : {ai_response}")

        messages.append({
            'role': 'assistant',
            'content': ai_response
        })

chat()

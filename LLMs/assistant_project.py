from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def save_history(messages):
    os.makedirs("data", exist_ok=True)
    with open("data/chat_history.json", "w") as f:
        json.dump(messages, f, indent=4)

    print('chat saved!')

def load_history():
    if os.path.exists("data/chat_history.json") and os.path.getsize("data/chat_history.json") > 0:
        with open("data/chat_history.json", "r") as f:
            return json.load(f)
    return None

def show_history(messages):
    print("=== chat history ===")
    for msg in messages:
        if msg['role'] == 'user':
            print(f'You : {msg['content']}')
        elif msg['role'] == 'assistant':
            print(f'AI : {msg['content']}')
    print("===============")


def main():
    print("=== personal Assistant ===")
    print("Commands : ")
    print("/history - show the history of chats")
    print("/clear - clear the chat")
    print("/quit - close the chat")

    messages = [{
        'role': 'system',
        'content': """You are a helpful assistant.
        You remember the conversation history and can communicate 
        fluently in both hindi and english."""
    }]
    
    old_history = load_history()
    if old_history:
        choice = input("Let's continue our previous conversation? (Y/n)")
        if choice.lower() == 'Y':
            messages = [messages[0]] + old_history[1:]
            print("Previous conversation loaded successfully")

    
    while True:

        user_input = input('You: ')

        if user_input.lower() == '/quit':
            save_history(messages)
            print('Bye')
            break

        elif user_input.lower() == '/history':
            show_history(messages)
            continue

        elif user_input.lower() == '/clear':
            messages = [messages[0]]
            save_history(messages)
            print("chat cleared")
            continue

        if not user_input.strip():
            continue

        messages.append(
            {'role': 'user',
             'content': user_input}
        )

        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )

        ai_response = response.choices[0].message.content

        print(f'AI : {ai_response}')

        messages.append(
            {
                'role': 'assistant',
                'content': ai_response
            }
        )
        save_history(messages)

main()
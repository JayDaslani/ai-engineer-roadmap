from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def ask(system_role, user_question):
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages = [
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': user_question}

        ],
        temperature = 0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

question = "How to build career in data science?"

print("=== Career Counselor ===")
print(ask("""You are a experienced career counselor 
          who guides Indian student. 
          Speak in Simple English.""",question))


print("=== Strict teacher ===")
print(ask("""Act as a strict teacher.
          Give direct and honest answer with no sugercoating.
          Always highlight the real challenges.""",question))

print("=== Data Scientist ===")
print(ask("""Act as a succesful Data Scientist with 5 years of industy experince.
          Share personal experiences and provide practical advise.""",question))

print("=== Interviewer ===")
print(ask("""Act as a technical interviewer hiring data science candidates. 
        Explain exactly what skills and qualities you loof for in a candidate.""",question))


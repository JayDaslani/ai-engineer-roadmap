

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

parser = StrOutputParser()
chain = template | llm | parser


history = []

def chat_with_memory(question):
    response = chain.invoke({
        "history": history,
        "question": question
    })
    
    
    history.append(HumanMessage(question))
    history.append(AIMessage(response))
    
    return response


print("=== Memory Test ===")

print("\nQ1:")
print(chat_with_memory("My name is Jay"))

print("\nQ2:")
print(chat_with_memory("I am live in Ahmedabad."))

print("\nQ3:")
print(chat_with_memory("What is my name ?"))

print("\nQ4:")
print(chat_with_memory("Where do i live ?"))


def interactive_chat():
    chat_history = history.copy()

    print("=== Interactive chat ===")
    print("Press quit for chat close")

    while True:
        user_input = input("You :")

        if user_input.lower() == 'quit':
            print('Bye')
            break

        response = chain.invoke({
            'history': chat_history,
            'question': user_input
        })


        history.append(HumanMessage(user_input))
        history.append(AIMessage(response))

        print(f'AI : {response}')

interactive_chat()


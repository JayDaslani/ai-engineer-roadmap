from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY')
)

response = llm.invoke("What is python ?")
print(response.content)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("My name is Jay")
]

response = llm.invoke(messages)
print(response.content)
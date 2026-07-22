from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY')
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} ho"),
    ("human", "{question}")
])

prompt = template.invoke({
    "role": "Python teacher",
    "question": "What is python ?"
})

response = llm.invoke(prompt)
print(response.content)
print(response.usage_metadata)

prompt2 = template.invoke({
    "role": "career counselor",
    "question": "How to build career in data science ?"
})

response2 = llm.invoke(prompt2)
print("=== career counselor")
print(response2.content)
print(response.usage_metadata)

prompt3 = template.invoke({
    'role': 'chef',
    'question': "How to make spiecy dal?"
})
response3 = llm.invoke(prompt3)
print("=== chef ===")
print(response3.content)
print(response3.usage_metadata)

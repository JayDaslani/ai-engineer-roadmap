from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY')
)

parser = StrOutputParser()

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}')
])

chain = template | llm | parser

response = chain.invoke({
    'question': 'What is python ?, define in 2 lines.'
})

print(response)
print(type(response))


json_parser = JsonOutputParser()

json_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a data extraction assistant. Always return valid json not text.'),
    ('human', '{question}')
])

json_chain = json_template | llm | json_parser

result2 = json_chain.invoke({
    "question": """
     Generate a JSON regarding python:
     {
         "name": "",
         "created_year": 0,
         "creator": "",
         "top_uses": [],
         "Type": ""
     }
     """
})

print("=== Json Output ===")
print(result2)
print(type(result2))
print(f"Name : {result2['name']}")
print(f"Creator: {result2['creator']}")
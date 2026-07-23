from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY')
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = template | llm | parser

response = chain.invoke({
    "question": "What is Python?"
})

print(response)

summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a summarizer. describe in 3 points"),
    ("human", "Tell me something about this {topic}")
])

summary_chain = summary_template | llm | parser


translate_template = ChatPromptTemplate.from_messages([
    ("system", "You are a translater. translate into hindi."),
    ("human", "{text}")
])

translate_chain = translate_template | llm | parser


result1 = summary_chain.invoke({
    "topic": "Machine Learning"
})
print("=== Summary ===")
print(result1)

result2 = translate_chain.invoke({
    "text": result1
})
print("== Hindi Translation ==")
print(result2)
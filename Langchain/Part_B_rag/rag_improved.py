from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
FILE_PATH = os.path.join(DATA_DIR, "jay_detailed.txt")
DB_DIR = os.path.join(DATA_DIR, "improved_rag_db")

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

loader = TextLoader(FILE_PATH, encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

vectorestore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=DB_DIR
)

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.1
)

retriever = vectorestore.as_retriever(
    search_kwargs={"k":4}
)

parser = StrOutputParser()

bad_template = ChatPromptTemplate.from_messages([
    ("system", "Answer using context : {context}"),
    ("human", "{question}")
])

good_template = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document assistant.
     
     STRICT RULES :
     1. Answer ONLY using the information provided in the context.
     2. If the information is not in the context - say EXACTLY: 'This information is not in the documents.'
     3. DO NOT add ANYTHING from your own knowledge.
     4. Provide short and precise answers.
     5. Use bullet points if there are multiple points. 
     
     Context : {context}"""),
     ("human", "{question}")
])

def compare_prompts(question):
    docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in docs])

    print(f"Question : {question}")

    bad_chain = bad_template | llm | parser
    bad_answer = bad_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Bad Prompt Answer: {bad_answer}")

    good_chain = good_template | llm | parser
    good_answer = good_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Good Prompt Answer: {good_answer}")

compare_prompts("Jay ki skills kya hain?")
compare_prompts("Jay ka favourite food kya hai?")

query_improve_template = ChatPromptTemplate([
    ("system", """You are a search query optimizer.
     Take the user's question and convert it into a better search query.
     
     RULES:
     1. Extracted keywords.
     2. Add synonyms.
     3. Return ONLY the query. No explantions."""),
     ("human", "Original query : {query}")
])

def improved_query_rag(question):
    print(f"Original : {question}")

    query_chain = query_improve_template | llm | parser

    better_query = query_chain.invoke({
        "query": question
    })
    print(f"Improved : {better_query}")

    docs = retriever.invoke(better_query)
    context = " ".join([doc.page_content for doc in docs])

    good_chain = good_template | llm | parser
    answer = good_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Answer: {answer}")

improved_query_rag("Jay kahaan hai?")
improved_query_rag("Jay kya jaanta hai?")
improved_query_rag("Jay kya karna chahta hai?")


print("=== Final Best RAG ===")

def best_rag(question):
    print(f"Question : {question}")

    query_chain = query_improve_template | llm | parser

    better_query = query_chain.invoke({
        "query": question
    })

    docs = retriever.invoke(better_query)

    context_parts = []

    for doc in docs:
        source = doc.metadata.get(
            'source', 'document'
        )
        context_parts.append(
            f"[Source: {source}]\n"
            f"{doc.page_content}"
        )

    context = " ".join(context_parts)

    good_chain = good_template | llm | parser
    answer = good_chain.invoke({
        "context": context,
        "question": question
    })

    print(f"Answer : {answer}")
    return answer

best_rag("Jay ke baare mein batao")
best_rag("Jay ka future plan kya hai?")
best_rag("Jay ne kaunsi cheezein seekhi hain?")

print("=== Best RAG chat ===")
print("'quit' - for chat close")

while True:
    q = input("You : ")
    if q.lower() == "quit":
        print("Bye!")
        break
    best_rag(q)


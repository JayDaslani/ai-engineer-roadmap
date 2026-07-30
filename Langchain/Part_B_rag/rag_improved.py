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

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

loader = TextLoader("../data/jay_detailed.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

vectorestore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="Langchain/data/improved_rag_db"
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
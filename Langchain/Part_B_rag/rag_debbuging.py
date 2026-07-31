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

BASE_DIR = "/Users/jaydasalani/Desktop/ai-engineer-roadmap"

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

loader = TextLoader(
    os.path.join(BASE_DIR, "Langchain/data/jay_detailed.txt")
)

documents = loader.load()

print("=== Problem 1: chunk size ===")
bad_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
bad_chunks = bad_splitter.split_documents(documents)
print(f"Bad chunks : {len(bad_chunks)}")

good_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
good_chunks = good_splitter.split_documents(documents)
print(f"Good chunks : {len(good_chunks)}")

print("Lesson: Zyada chunks = Better retrieval")

print("=== Problem 2 : k value ===")

vectorestore = Chroma.from_documents(
    documents=good_chunks,
    embedding=embeddings_model,
    persist_directory=os.path.join(
        BASE_DIR, "Langchain/data/debug_db"
    )
)

question = "Jay ki skills kya hain?"

bad_retriever = vectorestore.as_retriever(
    search_kwargs={"k":1}
)
bad_docs = bad_retriever.invoke(question)
print(f"k=1 - Retrieved : {len(bad_docs)} chunks")
print(f"Content : {bad_docs[0].page_content}")

good_retriever = vectorestore.as_retriever(
    search_kwargs={"k":4}
)
good_docs = good_retriever.invoke(question)
print(f"k=4 - Retrieved : {len(good_docs)} chunks")
for doc in good_docs:
    print(f"-> {doc.page_content}")

print("Lesson: k=3-5 usually best hai")

print("=== Problem 3 : Hallucination ===")
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.1
)

parser = StrOutputParser()

bad_template = ChatPromptTemplate.from_messages([
    ("system", "Be helpful. Context: {context}"),
    ("human", "{question}")
])

good_template = ChatPromptTemplate.from_messages([
    ("system", """STRICT RULES:
1. ONLY use provided context
2. NEVER add outside knowledge
3. If not in context — say exactly:
   "Ye information documents mein nahi hai"
4. Short precise answers only

Context:
{context}"""),
    ("human", "{question}")
])

def test_hallucination(question):
    docs = good_retriever.invoke(question)
    context = " ".join([d.page_content for d in docs])
    print(f"Questio : {question}")

    bad_chain = bad_template | llm | parser
    bad_ans = bad_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Bad: {bad_ans[:100]}")

    good_chain = good_template | llm | parser
    good_ans = good_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Good : {good_ans[:100]}")

test_hallucination("Jay ka favourite cricket team kaunsa hai?")
test_hallucination("Jay kaha rehta hai ?")

print("=== Problem 4. File Paths ===")

def check_path(path):
    if os.path.exists(path):
        print(f"File exsits: {path}")
    else:
        print(f"File not found : {path}")

check_path(
    os.path.join(BASE_DIR,
    "Langchain/data/jay_detailed.txt")
)
check_path(
    os.path.join(BASE_DIR,
    "Langchain/data/sample.pdf")
)

print("=== Problem 5: Empty context ===")

def safe_rag(question):
    print(f"Question : {question}")

    docs = good_retriever.invoke(question)

    if not docs:
        print("⚠️ Koi relevant docs nahi mile!")
        return
    
    context = " ".join([d.page_content for d in docs if d.page_content.strip()])

    if not context.strip():
        print("Context is empty")
        return 
    
    print(f"{len(docs)} chunks retrieved")

    good_chain = good_template | llm | parser
    answer = good_chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Answer : {answer}")

safe_rag("Jay ka naam kya hai?")
safe_rag("Jay ka favourite movie kya hai?")

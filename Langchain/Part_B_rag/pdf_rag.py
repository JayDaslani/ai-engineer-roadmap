from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "../data/sample.pdf"

print("1. Loading pdf...")
loader = PyMuPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Pages loaded : {len(pages)}")
print(f"First page preview: {pages[0].page_content[:200]}")


print("2. Creating chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(pages)
print(f"Total chunks : {len(chunks)}")

print("3. Loading embeddings...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("Ready!")

print("4. Storing in vectoredatabase..")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="Langchain/data/pdf_rag_db"
)
print(f"   Stored {len(chunks)} chunks!")

retriever = vectorstore.as_retriever(
    search_kwargs={"k":4}
)

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.1
)

template = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant.
     
     RULES:
     1. Answer ONLY based on the provided context.
     2. Mention the page number if available.
     3. If the information is not the context - state it clearly.
     4. Provide short and precise answers.
    Context : {context}"""),
    ("human", "{question}")  
])

parser = StrOutputParser()
chain = template | llm | parser

def ask_pdf(question, show_sources=False):
    print(f"Question: {question}")

    docs = retriever.invoke(question)

    context_parts = []
    for doc in docs:
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content
        context_parts.append(f"[Page {page}]: {content}")

    context = " ".join(context_parts)

    if show_sources:
        print("Sources:")
        for doc in docs:
            page = doc.metadata.get('page','N/A')
            print(f"  Page {page}: "
                  f"{doc.page_content[:80]}...")
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"Answer : {answer}")

print("=== Testing PDF RAG ===")

ask_pdf("What is this document about ?", show_sources=True)
ask_pdf("What skills are required for AI Engineer ?")
ask_pdf("What are the learning resources mentioned?")
ask_pdf("What is the roadmap for beginners ?")

print('=== pdf chat ===')
print(f"PDF : {PDF_PATH}")
print("Chat with you pdf")
print("'Source' - let's see source")
print("Press quit for close")

show_sources = False

while True:
    question = input("You: ")

    if question.lower() == 'quit':
        print('Bye!')
        break
    elif question.lower() == "sources":
        show_sources = not show_sources
        print(f"Sources: {'ON' if show_sources else 'OFF'}")
        continue

    ask_pdf(question, show_sources=show_sources)
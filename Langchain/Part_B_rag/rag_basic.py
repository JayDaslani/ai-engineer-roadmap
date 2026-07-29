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

print("=== Step 1: Indexing ===")

print("1. Document loading ...")
loader = TextLoader("../data/jay_info.txt")
documents = loader.load()
print(f"   Loaded: {len(documents)} document")

print("2. splitting into chunks ...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
print(f"   Chunks: {len(chunks)}")

print("3. Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)
print(" Model ready !")

print("4. Storing in VectorDatabase..")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="langchain/data/rag_db"
)
print(f"   Stored {len(chunks)} chunks!")
print("INDEXING COMPLETE!")

print("=== Step 2: RETRIEVAL + GENERATION ===")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

retriever = vectorstore.as_retriever(
    search_kwargs={'k':3}
)

template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
     Answer the question based on the context provide below.

    Context:
     {context}
     
     If the answer is not in the context,
     say: 'I don't have information about this'"""),
     ('human',"{question}")
])

parser = StrOutputParser()

def ask_rag(question):
    print(f"question : {question}")

    relevant_docs = retriever.invoke(question)

    context = " ".join([doc.page_content for doc in relevant_docs])

    print(f"Retrieved : {len(relevant_docs)} chunks")

    chain = template | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": question 
    })

    print(f"Answer : {answer}")
    return answer

#ask_rag('Where does Jay live ?')
#ask_rag("What skills does Jay have ?")
#ask_rag("What is Jay' goal ?")
#ask_rag("What kind of business does Jay want to start ?")

print("\n=== INTERACTIVE RAG ===")
print("Chat with your documents")
print("press 'quit' for chat close")

while True:
    question = input("You : ")

    if question.lower() == "quit":
        print("Bye!")
        break

    ask_rag(question)



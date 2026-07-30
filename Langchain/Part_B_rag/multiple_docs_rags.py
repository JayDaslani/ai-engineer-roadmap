from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Multiple Documents RAG ===")

print("1. Loading multiple files ...")
all_documents = []

# Dynamic Base Data Directory Path (To prevent FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

# Loader 1: Text File
loader1 = TextLoader(os.path.join(DATA_DIR, "jay_detailed.txt"), encoding="utf-8")
docs1 = loader1.load()
for doc in docs1:
    doc.metadata["source_name"] = "Jay Info"
all_documents.extend(docs1)

# Loader 2: PDF File
loader2 = PyPDFLoader(os.path.join(DATA_DIR, "sample.pdf"))
docs2 = loader2.load()
for doc in docs2:
    doc.metadata['source_name'] = "Ai engineer roadmap Info"
all_documents.extend(docs2)

# Loader 3: Text File
loader3 = TextLoader(os.path.join(DATA_DIR, "sample.txt"), encoding="utf-8")
docs3 = loader3.load()
for doc in docs3:
    doc.metadata['source_name'] = 'Sample Info'
all_documents.extend(docs3)

print(f"   Total documents loaded: {len(all_documents)}")
for doc in all_documents:
    print(f"   → {doc.metadata['source_name']}")

print("\n2. Creating chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)
chunks = splitter.split_documents(all_documents)
print(f"   Total chunks created: {len(chunks)}")

print("\n3. Loading Embedding Model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

print("4. Storing in Vector Database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=os.path.join(DATA_DIR, "multi_doc_db")
)
print(f"   Stored {len(chunks)} chunks successfully!")

# Setup LLM & Retriever
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.1
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Fixed .from_messages Syntax 👇
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that works with multiple documents.
     
     RULES:
     1. Answer based ONLY on the provided context.
     2. Specify which document the answer came from.
     3. If the information is in multiple documents - mention all of them.
     4. Be clear and precise.

     Context : {context}"""),
    ('human', "{question}")
])

parser = StrOutputParser()
chain = template | llm | parser

def ask_multi(question):
    print(f"\nQuestion : {question}")

    docs = retriever.invoke(question)

    context_parts = []
    sources = set()

    for doc in docs:
        source = doc.metadata.get(
            'source_name',
            doc.metadata.get('source', 'Unknown')
        )
        sources.add(source)
        context_parts.append(
            f"[From: {source}]\n{doc.page_content}"
        )

    context = "\n---\n".join(context_parts)

    print(f"Sources used: {', '.join(sources)}")

    answer = chain.invoke({
        "context": context,
        "question": question
    })
    print(f"Answer: {answer}\n")


print("\n=== Testing Multiple Docs ===")
ask_multi("Jay kaha rehta hai ?")
ask_multi("LangChain mein kya hota hai?")

print("=== Interactive Mode ===")
print("Chat with your multiple documents ('quit' to exit)")

while True:
    q = input("You : ")
    if q.lower() == 'quit':
        print('Bye!')
        break

    ask_multi(q)
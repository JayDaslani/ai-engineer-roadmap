from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

parser = StrOutputParser()

PERSIST_DIR = "./chroma_db"
NOTES_DIR = "./notes"

os.makedirs(NOTES_DIR, exist_ok=True)

def load_all_notes():
    documents = []
    try:
        txt_loader = DirectoryLoader(
         NOTES_DIR,
         glob="**/*.txt",
         loader_cls=TextLoader,
         loader_kwargs={'encoding': 'utf-8'}
     )
        documents.extend(txt_loader.load())
    except Exception as e:
        print(f"Text loading error: {e}")
      

    try:
        pdf_loader = DirectoryLoader(
            NOTES_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader  
       )
        documents.extend(pdf_loader.load())
    except Exception as e:
        print(f"PDF loading error: {e}")

    for doc in documents:
        file_path = doc.metadata.get('source', '')
        file_name = os.path.basename(file_path)
        doc.metadata['source_file'] = file_name

    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    raw_chunks = text_splitter.split_documents(documents)

    valid_chunks = []

    for doc in raw_chunks:
        content = str(doc.page_content) if doc.page_content is not None else ""
        cleaned = content.strip()
        if cleaned and len(cleaned) > 5:
            doc.page_content = cleaned
            valid_chunks.append(doc)

    return valid_chunks


def get_or_create_vectorstore():

    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0 :
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model
        )
    else:
        docs = load_all_notes()
        if not docs:
            return None
        chunks = create_chunks(docs)
        if chunks:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=PERSIST_DIR
            )
        else:
            vectorstore = None
    
    return vectorstore

def add_new_note(file, vectorstore):
    file_path = os.path.join(NOTES_DIR, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.name.endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        return "Unsupported file format!", vectorstore
    
    docs = loader.load()
    for doc in docs:
        doc.metadata['source_file'] = file.name

    chunks = create_chunks(docs)

    if chunks:
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=PERSIST_DIR
            )
        else:
            vectorstore.add_documents(chunks)

    return f"✅ '{file.name}' Successfully added to Knowledge Base!", vectorstore  

def ask_knowledge_base(question: str, vectorstore):
    if vectorstore is None:
        return "Your knowledge base is currently empty.Please add some notes first!", []
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    relevent_docs = retriever.invoke(question)

    if not relevent_docs:
        return "No information was found in the knowledge base on this topic.", []
    
    context = "\n\n".join([f"[Source: {doc.metadata.get('source_file', 'Note')}]: {doc.page_content}" for doc in relevent_docs])
    sources = list(set([doc.metadata.get('source_file', 'Unknown') for doc in relevent_docs]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the user's personal ai assistant. Provide a detailed and helpful answer to the user's question based on the personal notes/context provided below."),
        ("human", "Context (Personal Notes):\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    chain = prompt | llm | parser
    response = chain.invoke({"context": context, "question": question})

    return response, sources

def generate_quiz(topic: str, vectorstore):
    if vectorstore is None:
        return "The knowledge base is empty. A quiz cannot be generated."
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    relevent_docs = retriever.invoke(topic)

    context = "\n\n".join([doc.page_content for doc in relevent_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart tutor. Generate a topic-specific quiz for the user based on the provided notes/context."),
        ("human", "Topic: {topic}\n\nContext Notes:\n{context}\n\nTask: Generate 5 multiple choice question (MCQ) on this topic with options (A,B,C,D) and provide an Answer key at the end.")
    ])

    chain = prompt | llm | parser
    response = chain.invoke({'topic': topic, "context": context})

    return response

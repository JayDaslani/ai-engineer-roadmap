from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import tempfile
from typing import List, Tuple

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.2
)

parser = StrOutputParser()

def load_documents(files):
    all_documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name

        try:
            loader = PyPDFium2Loader(temp_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata['source_file'] = file.name
                all_documents.append(doc)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_documents



def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
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



def create_vectorstore(chunks):
    if not chunks:
        return None

    
    texts = [str(doc.page_content) for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    return vectorstore

def generate_summary(documents, llm_instance=llm):
    combined_text = "\n\n".join([f"[Source : {doc.metadata.get('source_file')}]\n{doc.page_content}" for doc in documents[:12]])

    prompt = ChatPromptTemplate.from_messages([
        ('system', "You are a helpful AI assistant. You need to provide a clear, structured, and insightful summary of the provide documents."),
        ("human", "{text}")

    ])
    chain = prompt | llm_instance | parser
    response = chain.invoke({"text" : combined_text[:10000]})
    return response


def extract_key_points(documents, llm_instance=llm):
    combined_text = "\n\n".join([doc.page_content for doc in documents[:12]])

    prompt = ChatPromptTemplate.from_messages([
        ('system', "You are a expert document analyst. Your task is to extract the most important bullet points and key insights fromn the documents."),
        ("human", "Extract the main key points and key takeways from these documents : {text}")
    ])

    chain = prompt | llm_instance | parser
    response = chain.invoke({"text": combined_text[:10000]})
    return response

def compare_documents(docs, llm_instance=llm):
    docs_by_file = {}
    for doc in docs:
        source = doc.metadata.get("source_file", "Unkown File")
        docs_by_file[source] = docs_by_file.get(source, "") + "\n" + doc.page_content

    if len(docs_by_file) < 2:
        return "Upload 2 different files for comparison."
    
    combined_sources = ""
    for file_name, content in docs_by_file.items():
        combined_sources += f"=== FILE NAME : {file_name} === {content[:4000]}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a comparative analyst. Your task is to compare the provided documents and outline their similarites, differences, and a conclusion."),
        ("human", "Compare these different documents in a structured way : {data}")
    ])
    chain = prompt | llm_instance | parser
    response = chain.invoke({"data" : combined_sources})
    return response

def ask_question(question, vectorstore):
    if not vectorstore:
        return "The vectorstore is not intialized", []
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([f"[File : {doc.metadata.get('source_file')}]: {doc.page_content}" for doc in relevant_docs])
    source = list(set([doc.metadata.get('source_file', 'Unkown') for doc in relevant_docs]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide an accurate answer to the user's question based on the context given below. If the information is not in the context, please state that."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    chain = prompt | llm | parser
    response = chain.invoke({"context": context, "question": question})
    return response, source
    


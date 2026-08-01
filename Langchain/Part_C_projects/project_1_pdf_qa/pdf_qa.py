from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
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

def load_pdf(pdf_path):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF File is not found at path : {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks, db_path=None):
    if db_path:
        vectorestore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=db_path
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model
        )
    return vectorestore

llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=os.getenv('GROQ_API_KEY'),
        temperature=0.1,
        max_tokens=1000
    )


def get_answer(question, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k':6}
    )

    docs = retriever.invoke(question)

    if not docs:
        return "Is document mein is sawal ka koi jawab nahi mila.", []
    
    context_parts = []
    sources = set()

    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        sources.add(page_num)
        context_parts.append(f"[Page {page_num}: {doc.page_content}]")

    context = "\n\n".join(context_parts)

    

    tempalete = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent PDF Q&A assistant.
         
         RULES:
1. Answer based on context
2. If partial info available —
   share what you know
3. Only say 'not found' if
   ZERO relevant info
4. Mention page numbers
5. Answer in same language
   as question

         
         Context: {context}"""),
         ("human","{question}")
    ])

    parser = StrOutputParser()

    chain = tempalete | llm | parser

    answer = chain.invoke({
        'context': context,
        "question": question
    })

    sorted_pages = sorted(list(sources))
    return answer, sorted_pages


PDF_PATH = "Langchain/Part_C_projects/project_1_pdf_qa/data/basic-economics.pdf"

print("Loading PDF ...")
docs = load_pdf(PDF_PATH)
print(f"pages : {len(docs)}")

print("Creating chunks ...")
chunks = create_chunks(docs)
print(f"Chunks : {len(chunks)}")

print("Creating Vectorstore...")
vs = create_vectorstore(
    chunks,
    db_path="Langchain/Part_C_projects/project_1_pdf_qa/data/db"

)
print("Ready!")

# Economics specific questions:

questions = [
    "What is inflation?",
    "What is GDP?",
    "What is market economy?",
    "What is price mechanism?",
    "What causes unemployment?"
]

for q in questions:
    answer, pages = get_answer(q, vs)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print(f"Pages: {pages}\n")


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

print("=== Advanced RAG ===")
loader = TextLoader("../data/jay_detailed.txt")
documents = loader.load()
print(f"Loaded: {len(documents)} document")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)
print(f"Chunks : {len(chunks)}")

print("Sample chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:{chunk.page_content}")

embeddings_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="Langchain/data/advanced_rag_db"
)
print(f"Stored {len(chunks)} chunks!")

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 4}
)

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.1
)

template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
     
     IMPORTANT RULES:
     1.Answer ONLY based on the provide context.
     2.If the answer is in the context - state it confidently.
     3.If the answer is not in the context - clearly state: 'This information is not the context.'
     4.Provide short and clear answer.

     Context: {context}"""),
     ("human", "{question}")
])

parser = StrOutputParser()
chain = template | llm | parser

def ask_rag(question, show_chunks=False):
    print(f"question : {question}")

    relevent_docs = retriever.invoke(question)

    context = " ".join([f"- {doc.page_content}" for doc in relevent_docs])

    if show_chunks:
        print(f"Reterived chunks :")
        for doc in relevent_docs:
            print(f" -> {doc.page_content}")

    answer = chain.invoke({
        'context': context,
        "question": question
    })

    print(f"Answer: {answer}")
    return answer

print("=== Testing ===")
'''
ask_rag('Jay kaha rehta hai ? ',show_chunks=True)
ask_rag('Jay ki skills kya hai ?')
ask_rag('Jay ka goal kya hai ?')
ask_rag('Jay kab graduate karega ?')
ask_rag('Jay ne kaunse project banaye ?')
ask_rag("Jay Pune kyun jaana chahta hai?")
'''


print("=== Interactive Mode ===")
print("Chat with your documents")
print('Press quit for close')

show = False

while True:
    question = input("You :")

    if question.lower() == "quit":
        print('Bye!')
        break
    elif question.lower() == "chunks":
        show = not show
        print(f"Chunks display : {show}")
        continue

    ask_rag(question, show_chunks=show)


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

documents = [
    Document(
        page_content="Python is a programming language",
        metadata={"category": "tech", "topic": "python"}
    ),
    Document(
        page_content='Langchain is a AI framework',
        metadata={"category": "tech", "topic": "langchain"}
    ),
    Document(
        page_content="Jay lives in Ahemdabad",
        metadata={"category": "personal", "topic": "location"}
    ),
    Document(
        page_content="Jay knows python and SQL",
        metadata={"category": "personal", "topic": "skills"}
    ),
    Document(
        page_content="Cricket is popular in India",
        metadata={"category": "sports", "topic": "cricket"}
    ),
    Document(
        page_content="Pune is an IT hub in India",
        metadata={"category": "city", "topic": "pune"}
    ),
    Document(
        page_content="Machine learning uses data",
        metadata={"category": "tech", "topic": "ml"}
    )

]
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory="LangChain/data/chroma_db2"
)

print(f"{len(documents)} documents stored!")

print("=== Basic Search ===")
results = vectorstore.similarity_search(
    "Python programming",
    k=2
)
for doc in results:
    print(f"-> {doc.page_content}")
    print(f" Metadata: {doc.metadata}")

print("=== Search with Score ===")
results = vectorstore.similarity_search_with_score(
    "AI and machine learning",
    k=3
)
for doc, score in results:
    print(f"Score : {score:.3f} -> {doc.page_content}")

print("=== Filter By Metadata ===")
results = vectorstore.similarity_search(
    "programming",
    k=3,
    filter={"category": "tech"}
)
print("Only tech category:")
for doc in results:
    print(f"-> {doc.page_content}")

print("=== Retriever ===")
retriever = vectorstore.as_retriever(
    search_kwargs={"k":2}
)
docs = retriever.invoke("Jay's skills ?")
for doc in docs:
    print(f"→ {doc.page_content}")
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import os


embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


client = chromadb.PersistentClient(
    path="LangChain/data/chroma_db"
)


collection = client.get_or_create_collection(
    name="my_collection"
)

print("ChromaDB setup complete!")
print(f"Collection: {collection.name}")

documents = [
    "Jay is a BTech AI/ML student",
    "Jay lives in Ahmedabad",
    "Jay wants to become AI Engineer",
    "Jay knows Python and SQL",
    "Jay wants to move to Pune",
    "Cricket is popular in India",
    "The weather in Mumbai is hot"
]

embeddings = embeddings_model.embed_documents(documents)

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"{len(documents)} documents added")
print(f"Total in DB: {collection.count()}")



def search(query, top_k=2):
    
    query_embedding = embeddings_model.embed_query(query)
    
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} results:")
    
    for doc, distance in zip(
        results['documents'][0],
        results['distances'][0]
    ):
        similarity = 1 - distance
        print(f"  {similarity:.3f} → {doc}")


search("Where does Jay live?")
search("What skills does Jay have?")
search("What is Jay's goal?")



print("\n=== Persistence Test ===")


new_client = chromadb.PersistentClient(
    path="LangChain/data/chroma_db"
)


same_collection = new_client.get_collection(
    name="my_collection"
)

print(f"Documents in DB: {same_collection.count()}")
print("Data persist ho gaya! ✅")


results = same_collection.query(
    query_embeddings=[
        embeddings_model.embed_query("Jay's city")
    ],
    n_results=1
)
print(f"Result: {results['documents'][0][0]}")
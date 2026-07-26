from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


texts = [
    "Python is a programming language",
    "Python coding tutorial",
    "I love eating apples",
    "Machine learning is amazing",
    "Deep learning neural networks"
]


embeddings = np.array(
    embeddings_model.embed_documents(texts)
)

print("=== Embeddings ===")
print(f"Total texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"\nFirst text: '{texts[0]}'")
print(f"First 5 numbers: {embeddings[0][:5]}")


print("\n=== Similarity Check ===")
query = "Python programming"
query_embedding = np.array(
    embeddings_model.embed_documents([query])
)

similarities = cosine_similarity(
    query_embedding,
    embeddings
)[0]

print(f"Query: '{query}'")
print("Similarity Scores:")
for text, score in zip(texts, similarities):
    print(f"{score:.3f} → {text}")

best_idx = np.argmax(similarities)
print(f"\nSabse similar:")
print(f"'{texts[best_idx]}'")
print(f"Score: {similarities[best_idx]:.3f}")


print("\n=== Real World Test ===")
documents = [
    "Jay is a BTech AI/ML student",
    "Jay lives in Ahmedabad",
    "Jay wants to become AI Engineer",
    "Jay knows Python and SQL",
    "Jay wants to move to Pune",
    "The weather in Mumbai is hot",
    "Cricket is popular in India"
]

doc_embeddings = np.array(
    embeddings_model.embed_documents(documents)
)

def find_similar(query, top_k=2):
    query_emb = np.array(
        embeddings_model.embed_documents([query])
    )
    scores = cosine_similarity(
        query_emb,
        doc_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f"\nQuery: '{query}'")
    print(f"Top {top_k} results:")
    for idx in top_indices:
        print(f"  {scores[idx]:.3f} → {documents[idx]}")

find_similar("Where does Jay live?")
find_similar("What does Jay want to do?")
find_similar("What programming skills Jay has?")
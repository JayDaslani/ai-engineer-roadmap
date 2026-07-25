
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
import os

loader = TextLoader('Langchain/data/sample.txt')
documents = loader.load()

print("Originl document")
print(f"Length : {len(documents[0].page_content)} character")
print(f"Content: {documents[0].page_content}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_documents(documents)

print(f"After splitting :")
print(f"Total chunks : {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"Content: {chunk.page_content}")
    print(f"Length : {len(chunk.page_content)}")

small_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10
)

small_chunks = small_splitter.split_documents(documents)
print(f"Total small chunks : {len(small_chunks)}")

large_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50
)
large_chunks = large_splitter.split_documents(documents)
print(f"Total large chunks : {len(large_chunks)}")


print("=== pdf splitting ===")

pdf_loader = PyPDFLoader('Langchain/data/sample.pdf')
pdf_docs = pdf_loader.load()

print(f"PDF pages : {len(pdf_docs)}")

pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

pdf_chunks = pdf_splitter.split_documents(pdf_docs)

print(f"Total chunks : {len(pdf_chunks)}")
print("First chunk :")
print(pdf_chunks[0].page_content)
print(f"Metadata : {pdf_chunks[0].metadata}")
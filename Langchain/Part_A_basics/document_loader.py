# LangChain/Part_A_Basics/06_document_loader.py

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
import os


print("=== Text File Loader ===")

loader = TextLoader("LangChain/data/sample.txt")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")
print(f"Content:\n{documents[0].page_content}")
print(f"Metadata: {documents[0].metadata}")


print("=== PDF Loader ===")

pdf_path = "Langchain/data/sample.pdf"

if os.path.exists(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    print(f"Pages loaded : {len(pdf_docs)}")

    print("Page 1 content : ")
    print(pdf_docs[0].page_content[:200])
    print(f"Metadata: {pdf_docs[0].metadata}")
else:
    print("Pdf file is not found")

print("=== Directory Loader ===")

dir_loader = DirectoryLoader(
    "Langchain/data/",
    glob="*.txt",
    loader_cls=TextLoader
)

all_docs = dir_loader.load()

print(f"Total documents : {len(all_docs)}")

for doc in all_docs:
    print(f"File : {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:50]}")
    print("----")



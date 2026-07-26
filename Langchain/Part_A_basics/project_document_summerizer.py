import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

def run_document_summerizer():
    print("=== Document Summerizer Project ===")

    loader = TextLoader('Langchain/data/sample.txt')
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) successfully.")

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitters.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    template = ChatPromptTemplate.from_messages([
        ('system', 'You are a text summarizer.Summarize this text.'),
        ('human', '{text}')
    ])

    parser = StrOutputParser()

    chain = template | llm | parser

    chunk_summerizers = []

    for idx, chunk in enumerate(chunks, start=1):

        summary = chain.invoke({'text':  chunk.page_content})
        print(f"Chunk {idx} Summarized.")

        chunk_summerizers.append({
            'chunk_id': idx,
            'original_text_snippet': chunk.page_content[:100]+'...',
            "summary": summary
        })

    print("Combining all chunk summaries for Final Summary...")

    all_summaries_text = "\n".join([item['summary'] for item in chunk_summerizers])

    final_template = ChatPromptTemplate.from_messages([
        ('system', 'Summaries of different chunks are provided below. Combine them to create a single structured final summary.'),
        ('human', '{combined_text}')
    ])

    final_chain = final_template | llm | parser
    final_summary = final_chain.invoke({'combined_text': all_summaries_text})

    print("Final Combined Summary Generated!")

    output_data = {
        'source_file': 'Langchain/data/sample.txt',
        'total_chunks': len(chunks),
        'chunk_summaries': chunk_summerizers,
        'final_summary': final_summary
    }

    os.makedirs('data',exist_ok=True)
    json_path = 'data/summary_result.json'

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to '{json_path}'!")

if __name__ == "__main__":
    run_document_summerizer()
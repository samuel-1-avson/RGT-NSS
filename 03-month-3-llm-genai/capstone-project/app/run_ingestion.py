import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.ingestion import load_documents, split_documents
from app.rag.retrieval import add_documents_to_db

def main():
    print("Starting document ingestion...")
    try:
        docs = load_documents()
        chunks = split_documents(docs)
        add_documents_to_db(chunks)
        print("Ingestion complete!")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    main()

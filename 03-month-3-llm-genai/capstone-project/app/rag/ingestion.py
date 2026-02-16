import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "documents")

def load_documents() -> List[Document]:
    """Load markdown documents from the data directory."""
    if not os.path.exists(DOCS_DIR):
        raise FileNotFoundError(f"Documents directory not found at {DOCS_DIR}")
        
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    d = load_documents()
    c = split_documents(d)

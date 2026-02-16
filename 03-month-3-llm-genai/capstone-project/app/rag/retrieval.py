import os
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

# Use a local directory for persistence
FAISS_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "faiss_db")

def get_vector_store() -> FAISS:
    """Initialize or load the FAISS vector store."""
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    
    if os.path.exists(FAISS_DB_DIR):
        try:
            return FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}, creating new one.")
            # Fallback to create empty if load fails
            # Note: FAISS requires at least one doc to initialize, but we can't create empty.
            # We'll handle this in add_documents_to_db
            return None
    return None

def add_documents_to_db(chunks: List[Document]):
    """Add document chunks to the vector database."""
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    
    if os.path.exists(FAISS_DB_DIR):
        vector_store = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        
    vector_store.save_local(FAISS_DB_DIR)
    print(f"Saved {len(chunks)} chunks to FAISS at {FAISS_DB_DIR}")

def retrieve_context(query: str, k: int = 4) -> List[Document]:
    """Retrieve relevant documents for a query."""
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    if not os.path.exists(FAISS_DB_DIR):
        return []
        
    vector_store = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_store.similarity_search(query, k=k)

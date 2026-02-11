"""
Vector Store Module for RAG Pipeline

Implements vector storage using FAISS and Chroma.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector stores for document retrieval.
    
    Supports:
    - FAISS (local, in-memory)
    - Chroma (persistent)
    """
    
    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Embedding model to use (default: OpenAIEmbeddings)
        """
        self.embeddings = embedding_model or OpenAIEmbeddings()
        self.vectorstore = None
        self.store_type = None
        logger.info("VectorStoreManager initialized")
    
    def create_faiss_index(
        self, 
        documents: List[Document]
    ) -> FAISS:
        """
        Create a FAISS index from documents.
        
        Args:
            documents: Documents to index
            
        Returns:
            FAISS vector store
        """
        logger.info(f"Creating FAISS index with {len(documents)} documents")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        self.store_type = 'faiss'
        
        logger.info("FAISS index created successfully")
        return self.vectorstore
    
    def create_chroma_index(
        self,
        documents: List[Document],
        persist_directory: str = "../data/chroma_db",
        collection_name: str = "documents"
    ) -> Chroma:
        """
        Create a Chroma index from documents.
        
        Args:
            documents: Documents to index
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            
        Returns:
            Chroma vector store
        """
        logger.info(f"Creating Chroma index with {len(documents)} documents")
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        self.store_type = 'chroma'
        
        # Persist the database
        self.vectorstore.persist()
        
        logger.info(f"Chroma index created and persisted to {persist_directory}")
        return self.vectorstore
    
    def load_faiss_index(self, index_path: str) -> Optional[FAISS]:
        """
        Load a saved FAISS index.
        
        Args:
            index_path: Path to the saved index
            
        Returns:
            FAISS vector store or None if not found
        """
        if not os.path.exists(index_path):
            logger.warning(f"FAISS index not found at {index_path}")
            return None
        
        logger.info(f"Loading FAISS index from {index_path}")
        
        self.vectorstore = FAISS.load_local(
            folder_path=index_path,
            embeddings=self.embeddings
        )
        self.store_type = 'faiss'
        
        logger.info("FAISS index loaded successfully")
        return self.vectorstore
    
    def load_chroma_index(
        self,
        persist_directory: str = "../data/chroma_db"
    ) -> Optional[Chroma]:
        """
        Load a saved Chroma index.
        
        Args:
            persist_directory: Directory where database is persisted
            
        Returns:
            Chroma vector store or None if not found
        """
        if not os.path.exists(persist_directory):
            logger.warning(f"Chroma index not found at {persist_directory}")
            return None
        
        logger.info(f"Loading Chroma index from {persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.store_type = 'chroma'
        
        logger.info("Chroma index loaded successfully")
        return self.vectorstore
    
    def save_faiss_index(self, index_path: str) -> None:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Raises:
            ValueError: If no FAISS index exists
        """
        if self.vectorstore is None or self.store_type != 'faiss':
            raise ValueError("No FAISS index to save")
        
        os.makedirs(index_path, exist_ok=True)
        self.vectorstore.save_local(index_path)
        logger.info(f"FAISS index saved to {index_path}")
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever from the vector store.
        
        Args:
            search_kwargs: Search configuration (e.g., {"k": 5})
            
        Returns:
            Retriever instance
            
        Raises:
            ValueError: If no vector store exists
        """
        if self.vectorstore is None:
            raise ValueError("No vector store initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of matching documents
        """
        if self.vectorstore is None:
            raise ValueError("No vector store initialized")
        
        logger.info(f"Searching for: '{query}' (k={k})")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Found {len(results)} matches")
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store.
        
        Args:
            documents: Documents to add
        """
        if self.vectorstore is None:
            raise ValueError("No vector store initialized")
        
        logger.info(f"Adding {len(documents)} documents to index")
        
        if self.store_type == 'chroma':
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
        else:
            self.vectorstore.add_documents(documents)
        
        logger.info("Documents added successfully")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        stats = {
            "store_type": self.store_type,
            "status": "active"
        }
        
        if self.store_type == 'chroma':
            try:
                count = self.vectorstore._collection.count()
                stats["document_count"] = count
            except Exception as e:
                stats["error"] = str(e)
        
        return stats


def compare_vector_stores(
    documents: List[Document],
    queries: List[str],
    embedding_model: Optional[OpenAIEmbeddings] = None
) -> Dict[str, Any]:
    """
    Compare FAISS and Chroma performance.
    
    Args:
        documents: Documents to index
        queries: Test queries
        embedding_model: Embedding model to use
        
    Returns:
        Comparison results
    """
    import time
    
    results = {}
    embeddings = embedding_model or OpenAIEmbeddings()
    
    # FAISS
    logger.info("Testing FAISS...")
    faiss_manager = VectorStoreManager(embeddings)
    start = time.time()
    faiss_manager.create_faiss_index(documents)
    faiss_build_time = time.time() - start
    
    faiss_results = []
    for query in queries:
        start = time.time()
        docs = faiss_manager.similarity_search(query, k=3)
        faiss_results.append({
            "query": query,
            "time": time.time() - start,
            "results": len(docs)
        })
    
    results['faiss'] = {
        'build_time': faiss_build_time,
        'queries': faiss_results
    }
    
    # Chroma
    logger.info("Testing Chroma...")
    chroma_manager = VectorStoreManager(embeddings)
    start = time.time()
    chroma_manager.create_chroma_index(documents, persist_directory="../data/chroma_compare")
    chroma_build_time = time.time() - start
    
    chroma_results = []
    for query in queries:
        start = time.time()
        docs = chroma_manager.similarity_search(query, k=3)
        chroma_results.append({
            "query": query,
            "time": time.time() - start,
            "results": len(docs)
        })
    
    results['chroma'] = {
        'build_time': chroma_build_time,
        'queries': chroma_results
    }
    
    return results


def main():
    """Example usage."""
    # Example: Create a FAISS index
    # from document_loader import DocumentLoader
    # loader = DocumentLoader()
    # docs = loader.load_file("../data/sample.pdf")
    # chunks = loader.chunk_documents(docs)
    # 
    # manager = VectorStoreManager()
    # manager.create_faiss_index(chunks)
    
    print("VectorStoreManager ready")


if __name__ == "__main__":
    main()

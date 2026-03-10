"""Retrievers implementation for document search."""

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_ollama import OllamaLLM
from typing import List, Optional
import os
from app.config import config
from app.instrumentation import log_retrieval


class DocumentRetriever:
    """Handles document loading, processing, and retrieval."""
    
    def __init__(self):
        """Initialize embeddings and vector store."""
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.vector_store = None
        self._ensure_chroma_dir()
    
    def _ensure_chroma_dir(self):
        """Ensure ChromaDB directory exists."""
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    
    def load_documents(self, documents: List[Document]) -> None:
        """Load and index documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        
        # Persist the vector store
        self.vector_store.persist()
        
        print(f"Loaded {len(documents)} documents into {len(chunks)} chunks")
    
    def load_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Load documents from raw texts.
        
        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
        """
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.load_documents(documents)
    
    @log_retrieval
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return (default from config)
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            # Try to load existing vector store
            try:
                self.vector_store = Chroma(
                    persist_directory=config.CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                )
            except Exception as e:
                raise ValueError(f"No vector store available: {e}")
        
        k = top_k or config.TOP_K_RETRIEVAL
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get a LangChain retriever object.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            Configured retriever
        """
        if self.vector_store is None:
            try:
                self.vector_store = Chroma(
                    persist_directory=config.CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                )
            except Exception as e:
                raise ValueError(f"No vector store available: {e}")
        
        kwargs = search_kwargs or {"k": config.TOP_K_RETRIEVAL}
        return self.vector_store.as_retriever(search_kwargs=kwargs)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add more documents to the existing vector store.
        
        Args:
            documents: New documents to add
        """
        if self.vector_store is None:
            self.load_documents(documents)
        else:
            chunks = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()


def create_compression_retriever(base_retriever, llm_model: str = None):
    """Create a contextual compression retriever for better results.
    
    Args:
        base_retriever: The base retriever to wrap
        llm_model: Optional LLM model name
        
    Returns:
        ContextualCompressionRetriever
    """
    llm = OllamaLLM(
        model=llm_model or config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    
    return compression_retriever

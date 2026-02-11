"""
Document Loader Module for RAG Pipeline

Handles loading and preprocessing of various document formats.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Handles loading and initial processing of documents.
    
    Supports formats:
    - .txt (plain text)
    - .pdf (PDF documents)
    - Directory of documents
    """
    
    def __init__(self):
        """Initialize the document loader."""
        self.supported_extensions = {'.txt', '.pdf', '.md'}
        logger.info("DocumentLoader initialized")
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """
        Load a plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s)")
        return documents
    
    def load_pdf_file(self, file_path: str) -> List[Document]:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        logger.info(f"Loading PDF file: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} page(s)")
        return documents
    
    def load_directory(
        self, 
        directory_path: str, 
        glob_pattern: str = "**/*"
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            glob_pattern: Glob pattern for file matching
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading documents from: {directory_path}")
        
        documents = []
        dir_path = Path(directory_path)
        
        for file_path in dir_path.glob(glob_pattern):
            if file_path.suffix.lower() in self.supported_extensions:
                try:
                    if file_path.suffix.lower() == '.txt':
                        docs = self.load_text_file(str(file_path))
                    elif file_path.suffix.lower() == '.pdf':
                        docs = self.load_pdf_file(str(file_path))
                    elif file_path.suffix.lower() == '.md':
                        docs = self.load_text_file(str(file_path))
                    else:
                        continue
                    
                    # Add source to metadata
                    for doc in docs:
                        doc.metadata['source_file'] = str(file_path)
                    
                    documents.extend(docs)
                    logger.info(f"  ✓ Loaded: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Error loading {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a file based on its extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        
        if ext == '.txt' or ext == '.md':
            return self.load_text_file(file_path)
        elif ext == '.pdf':
            return self.load_pdf_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use
            
        Returns:
            List of document chunks
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        logger.info(
            f"Chunking {len(documents)} documents "
            f"(size={chunk_size}, overlap={chunk_overlap})"
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Preserve source metadata
        for chunk in chunks:
            if 'source' not in chunk.metadata and 'source_file' in chunk.metadata:
                chunk.metadata['source'] = chunk.metadata['source_file']
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Log chunk size statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        logger.info(f"Chunk sizes - Min: {min(chunk_lengths)}, "
                   f"Max: {max(chunk_lengths)}, "
                   f"Avg: {sum(chunk_lengths)/len(chunk_lengths):.0f}")
        
        return chunks
    
    def compare_chunk_sizes(
        self,
        documents: List[Document],
        chunk_sizes: List[int] = [500, 1000, 1500]
    ) -> dict:
        """
        Compare different chunk sizes.
        
        Args:
            documents: Documents to chunk
            chunk_sizes: List of chunk sizes to compare
            
        Returns:
            Dictionary with results for each chunk size
        """
        results = {}
        
        for size in chunk_sizes:
            overlap = int(size * 0.2)  # 20% overlap
            chunks = self.chunk_documents(documents, chunk_size=size, chunk_overlap=overlap)
            
            results[size] = {
                'num_chunks': len(chunks),
                'avg_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks),
                'sample_chunks': chunks[:3]  # First 3 chunks for inspection
            }
            
            logger.info(f"Chunk size {size}: {len(chunks)} chunks created")
        
        return results


def main():
    """Example usage."""
    loader = DocumentLoader()
    
    # Example: Load a directory
    # documents = loader.load_directory("../data/documents")
    
    # Example: Compare chunk sizes
    # results = loader.compare_chunk_sizes(documents)
    
    print("DocumentLoader ready")


if __name__ == "__main__":
    main()

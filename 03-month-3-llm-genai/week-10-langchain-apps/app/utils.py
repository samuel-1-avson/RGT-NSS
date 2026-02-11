"""
Utility functions for the LangChain Q&A application.

This module provides helper functions for document processing,
embedding management, and common operations.
"""

import os
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file for change detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
        
    Example:
        >>> hash = calculate_file_hash("document.pdf")
        >>> print(hash)
        'a1b2c3d4...'
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_document_metadata(file_path: str) -> Dict:
    """
    Extract metadata from a document file.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary containing file metadata
        
    Example:
        >>> meta = get_document_metadata("doc.pdf")
        >>> print(meta['filename'])
        'doc.pdf'
    """
    path = Path(file_path)
    stats = os.stat(file_path)
    
    return {
        "filename": path.name,
        "filepath": str(path.absolute()),
        "file_type": path.suffix.lower(),
        "file_size_bytes": stats.st_size,
        "last_modified": stats.st_mtime,
        "file_hash": calculate_file_hash(file_path)
    }


def create_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> object:
    """
    Create a text splitter based on type.
    
    Args:
        splitter_type: Type of splitter ('recursive', 'character', 'token')
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        **kwargs: Additional arguments for the splitter
        
    Returns:
        Configured text splitter instance
        
    Example:
        >>> splitter = create_text_splitter("recursive", chunk_size=500)
        >>> chunks = splitter.split_documents(docs)
    """
    splitters = {
        "recursive": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "token": TokenTextSplitter
    }
    
    if splitter_type not in splitters:
        raise ValueError(
            f"Unknown splitter type: {splitter_type}. "
            f"Available: {list(splitters.keys())}"
        )
    
    splitter_class = splitters[splitter_type]
    
    if splitter_type == "token":
        return splitter_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    else:
        return splitter_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            **kwargs
        )


def split_documents_with_metadata(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents while preserving and enhancing metadata.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents with enhanced metadata
        
    Example:
        >>> docs = loader.load()
        >>> chunks = split_documents_with_metadata(docs, chunk_size=500)
        >>> print(chunks[0].metadata['chunk_index'])
        0
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enhance metadata with chunk information
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk.page_content)
        })
    
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def format_documents_for_context(documents: List[Document]) -> str:
    """
    Format a list of documents into a context string.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Formatted context string
        
    Example:
        >>> context = format_documents_for_context(retrieved_docs)
        >>> print(context[:200])
    """
    formatted = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        
        formatted.append(
            f"[Document {i} | Source: {source}, Page: {page}]\n"
            f"{doc.page_content}\n"
        )
    
    return "\n---\n".join(formatted)


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a text.
    
    This is a rough estimate (4 characters ≈ 1 token for English).
    For precise counts, use tiktoken.
    
    Args:
        text: Text to estimate
        model: Model name (for context window info)
        
    Returns:
        Estimated token count
        
    Example:
        >>> estimate_tokens("Hello world", "gpt-3.5-turbo")
        3
    """
    # Rough estimate: 4 characters per token for English
    return len(text) // 4


def check_context_window(
    text: str,
    model: str = "gpt-3.5-turbo",
    buffer: int = 500
) -> Tuple[bool, int, int]:
    """
    Check if text fits within model's context window.
    
    Args:
        text: Text to check
        model: Model name
        buffer: Safety buffer to leave room for response
        
    Returns:
        Tuple of (fits_in_window, token_count, max_tokens)
        
    Example:
        >>> fits, count, max_tok = check_context_window(long_text)
        >>> if not fits:
        ...     print(f"Text too long: {count} > {max_tok}")
    """
    context_windows = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000
    }
    
    max_tokens = context_windows.get(model, 4096)
    token_count = estimate_tokens(text)
    fits = (token_count + buffer) <= max_tokens
    
    return fits, token_count, max_tokens


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    truncation_side: str = "right"
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        truncation_side: 'left' or 'right' side to truncate from
        
    Returns:
        Truncated text
        
    Example:
        >>> short = truncate_to_token_limit(long_text, 1000)
    """
    # Rough conversion: tokens * 4 = characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    if truncation_side == "left":
        return "..." + text[-(max_chars - 3):]
    else:
        return text[:max_chars - 3] + "..."


def merge_chunked_documents(
    chunks: List[Document],
    separator: str = "\n\n"
) -> Document:
    """
    Merge multiple document chunks back into a single document.
    
    Args:
        chunks: List of Document chunks
        separator: Separator between chunks
        
    Returns:
        Merged Document
        
    Example:
        >>> merged = merge_chunked_documents(chunks)
        >>> print(len(merged.page_content))
    """
    # Sort by chunk index if available
    sorted_chunks = sorted(
        chunks,
        key=lambda x: x.metadata.get("chunk_index", 0)
    )
    
    merged_content = separator.join(
        chunk.page_content for chunk in sorted_chunks
    )
    
    # Merge metadata
    merged_metadata = {
        "source": chunks[0].metadata.get("source", "Unknown"),
        "total_chunks": len(chunks),
        "merged": True
    }
    
    return Document(page_content=merged_content, metadata=merged_metadata)


def chunk_statistics(chunks: List[Document]) -> Dict:
    """
    Calculate statistics about document chunks.
    
    Args:
        chunks: List of Document chunks
        
    Returns:
        Dictionary of statistics
        
    Example:
        >>> stats = chunk_statistics(chunks)
        >>> print(f"Avg chunk size: {stats['avg_chunk_size']}")
    """
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_chars": sum(sizes),
        "avg_chunk_size": sum(sizes) / len(sizes) if sizes else 0,
        "min_chunk_size": min(sizes) if sizes else 0,
        "max_chunk_size": max(sizes) if sizes else 0,
        "median_chunk_size": sorted(sizes)[len(sizes) // 2] if sizes else 0
    }


def get_embedding_cost_estimate(
    texts: List[str],
    model: str = "text-embedding-ada-002"
) -> Dict:
    """
    Estimate the cost of creating embeddings.
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        
    Returns:
        Cost estimate dictionary
        
    Example:
        >>> cost = get_embedding_cost_estimate(chunks)
        >>> print(f"Estimated cost: ${cost['estimated_cost_usd']:.4f}")
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "text-embedding-ada-002": 0.0001,
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013
    }
    
    total_tokens = sum(estimate_tokens(text) for text in texts)
    price_per_1k = pricing.get(model, 0.0001)
    estimated_cost = (total_tokens / 1000) * price_per_1k
    
    return {
        "model": model,
        "total_tokens": total_tokens,
        "price_per_1k_tokens": price_per_1k,
        "estimated_cost_usd": estimated_cost,
        "text_count": len(texts)
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
        
    Example:
        >>> safe = sanitize_filename("file with spaces & special!chars.pdf")
        >>> print(safe)
        'file_with_spaces___special_chars.pdf'
    """
    # Replace spaces and special characters
    sanitized = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in filename
    )
    return sanitized


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
        
    Example:
        >>> path = ensure_directory("./data/processed")
        >>> print(path.exists())
        True
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Token counter using tiktoken if available
try:
    import tiktoken
    
    def count_tokens_precise(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Precisely count tokens using tiktoken.
        
        Args:
            text: Text to count
            model: Model name for encoding
            
        Returns:
            Exact token count
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    # Replace estimate with precise count if tiktoken available
    def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        return count_tokens_precise(text, model)
        
except ImportError:
    logger.info("tiktoken not installed, using approximate token counts")

"""
End-to-End RAG System for Capstone Project

A production-ready RAG implementation with evaluation.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Add paths for week 11 and 12 modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week-11-rag-vector-db'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'week-12-evaluation'))

from rag.document_loader import DocumentLoader
from rag.vector_store import VectorStoreManager
from evaluation.ragas_evaluator import RAGEvaluator, create_evaluation_dataset_from_qa_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_type: str = "chroma"  # "faiss" or "chroma"
    persist_directory: str = "../data/vectorstore"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0
    top_k: int = 5


class RAGSystem:
    """
    End-to-End RAG System.
    
    Combines document loading, vector storage, retrieval, and generation
    with built-in evaluation capabilities.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: RAG configuration (uses defaults if None)
        """
        self.config = config or RAGConfig()
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.vector_manager = VectorStoreManager()
        self.evaluator = RAGEvaluator()
        
        self.qa_chain = None
        
        logger.info("RAGSystem initialized")
    
    def ingest_documents(
        self,
        source_path: str,
        source_type: str = "auto"
    ) -> List[Document]:
        """
        Ingest documents from file or directory.
        
        Args:
            source_path: Path to file or directory
            source_type: "file", "directory", or "auto"
            
        Returns:
            List of processed document chunks
        """
        logger.info(f"Ingesting documents from: {source_path}")
        
        # Load documents
        if source_type == "directory" or (source_type == "auto" and os.path.isdir(source_path)):
            documents = self.document_loader.load_directory(source_path)
        else:
            documents = self.document_loader.load_file(source_path)
        
        # Chunk documents
        chunks = self.document_loader.chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        logger.info(f"Ingested {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_index(self, chunks: List[Document]) -> None:
        """
        Create vector index from document chunks.
        
        Args:
            chunks: Document chunks to index
        """
        logger.info(f"Creating {self.config.vector_store_type} index")
        
        if self.config.vector_store_type == "faiss":
            self.vector_manager.create_faiss_index(chunks)
        else:
            self.vector_manager.create_chroma_index(
                chunks,
                persist_directory=self.config.persist_directory
            )
        
        logger.info("Index created successfully")
    
    def initialize_qa_chain(self) -> None:
        """
        Initialize the QA chain with retriever.
        """
        logger.info("Initializing QA chain")
        
        # Get retriever
        retriever = self.vector_manager.get_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        
        # Create LLM
        llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        
        logger.info("QA chain initialized")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call initialize_qa_chain() first.")
        
        logger.info(f"Processing query: {question}")
        start_time = datetime.now()
        
        result = self.qa_chain({"query": question})
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                }
                for doc in result.get("source_documents", [])
            ],
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate(
        self,
        test_questions: List[str],
        ground_truths: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system performance.
        
        Args:
            test_questions: List of test questions
            ground_truths: Optional ground truth answers
            output_path: Optional path to save report
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation with {len(test_questions)} questions")
        
        # Generate answers for test questions
        qa_data = create_evaluation_dataset_from_qa_chain(
            self.qa_chain,
            test_questions,
            ground_truths
        )
        
        # Prepare dataset
        dataset = self.evaluator.prepare_dataset(
            questions=qa_data["questions"],
            answers=qa_data["answers"],
            contexts=qa_data["contexts"],
            ground_truths=qa_data.get("ground_truths")
        )
        
        # Run evaluation
        results = self.evaluator.evaluate(dataset)
        
        # Generate report
        report = self.evaluator.generate_report(results, output_path)
        
        logger.info("Evaluation completed")
        
        return {
            "metrics": results,
            "report": report,
            "num_questions": len(test_questions)
        }
    
    def run_pipeline(
        self,
        source_path: str,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run complete pipeline: ingest, index, query.
        
        Args:
            source_path: Path to documents
            queries: List of questions
            
        Returns:
            List of results
        """
        # Ingest and index
        chunks = self.ingest_documents(source_path)
        self.create_index(chunks)
        self.initialize_qa_chain()
        
        # Process queries
        results = []
        for query in queries:
            result = self.query(query)
            results.append(result)
        
        return results


def main():
    """Example usage."""
    # Initialize system
    config = RAGConfig(
        chunk_size=1000,
        vector_store_type="chroma",
        model_name="gpt-3.5-turbo"
    )
    
    rag = RAGSystem(config)
    
    # Example: Process documents and answer questions
    # chunks = rag.ingest_documents("../data/documents")
    # rag.create_index(chunks)
    # rag.initialize_qa_chain()
    # result = rag.query("What is the main topic?")
    # print(result["answer"])
    
    print("RAGSystem ready")


if __name__ == "__main__":
    main()

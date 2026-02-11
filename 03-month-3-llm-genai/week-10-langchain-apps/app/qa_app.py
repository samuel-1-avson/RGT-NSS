"""
Q&A Application using LangChain

This module implements a Question-Answering system over PDF documents
using LangChain, OpenAI embeddings, and Chroma vector store.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentQASystem:
    """
    A Question-Answering system for PDF documents.
    
    This class handles document loading, text splitting, vector storage,
    and question answering using LangChain and OpenAI.
    
    Attributes:
        vectorstore: Chroma vector store instance
        qa_chain: RetrievalQA chain instance
        persist_dir: Directory for vector store persistence
    """
    
    def __init__(self, persist_dir: str = "../data/chroma_db"):
        """
        Initialize the QA system.
        
        Args:
            persist_dir: Directory to persist vector store
        """
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = OpenAIEmbeddings()
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"DocumentQASystem initialized with persist_dir: {persist_dir}")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        
        return documents
    
    def split_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split document chunks
        """
        logger.info(f"Splitting documents with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Chroma vector store instance
        """
        logger.info("Creating vector store...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        # Persist the vector store
        self.vectorstore.persist()
        logger.info(f"Vector store created and persisted to {self.persist_dir}")
        
        return self.vectorstore
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Load an existing vector store.
        
        Returns:
            Chroma vector store instance or None if not found
        """
        if not os.path.exists(self.persist_dir):
            logger.warning(f"No existing vector store found at {self.persist_dir}")
            return None
        
        logger.info(f"Loading existing vector store from {self.persist_dir}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        logger.info("Vector store loaded successfully")
        return self.vectorstore
    
    def create_qa_chain(self, model_name: str = "gpt-3.5-turbo") -> RetrievalQA:
        """
        Create the QA chain.
        
        Args:
            model_name: OpenAI model to use
            
        Returns:
            RetrievalQA chain instance
            
        Raises:
            ValueError: If vector store hasn't been created
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        logger.info(f"Creating QA chain with model: {model_name}")
        
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=30
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            verbose=False
        )
        
        logger.info("QA chain created successfully")
        return self.qa_chain
    
    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary containing answer and metadata
            
        Raises:
            ValueError: If QA chain hasn't been created
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")
        
        logger.info(f"Processing question: {question}")
        start_time = datetime.now()
        
        try:
            result = self.qa_chain({"query": question})
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "processing_time": processing_time,
                "timestamp": end_time.isoformat()
            }
            
            logger.info(f"Answer generated in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 1000) -> None:
        """
        Complete pipeline: Load PDF, split, create vector store, and setup QA.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
        """
        # Load and process PDF
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents, chunk_size=chunk_size)
        
        # Create vector store
        self.create_vectorstore(chunks)
        
        # Create QA chain
        self.create_qa_chain()
        
        logger.info("PDF processing complete. Ready for questions.")


def main():
    """Example usage of the QA system."""
    # Initialize system
    qa_system = DocumentQASystem()
    
    # Process a PDF (example)
    pdf_path = "../data/sample_document.pdf"
    
    if os.path.exists(pdf_path):
        qa_system.process_pdf(pdf_path)
        
        # Ask questions
        questions = [
            "What is the main topic of this document?",
            "What are the key findings?",
            "Can you summarize the conclusion?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            try:
                response = qa_system.ask(question)
                print(f"A: {response['answer']}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print(f"PDF not found: {pdf_path}")
        print("Please place a PDF file in the data directory.")


if __name__ == "__main__":
    main()

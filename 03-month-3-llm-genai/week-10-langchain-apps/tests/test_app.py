"""
Unit Tests for LangChain Q&A Application

Run with: python -m pytest test_app.py -v
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from qa_app import DocumentQASystem


class TestDocumentQASystem(unittest.TestCase):
    """Test cases for DocumentQASystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.qa_system = DocumentQASystem(persist_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.qa_system)
        self.assertEqual(self.qa_system.persist_dir, self.temp_dir)
        self.assertIsNone(self.qa_system.vectorstore)
        self.assertIsNone(self.qa_system.qa_chain)
    
    def test_load_pdf_file_not_found(self):
        """Test loading non-existent PDF."""
        with self.assertRaises(FileNotFoundError):
            self.qa_system.load_pdf("/nonexistent/path.pdf")
    
    @patch('qa_app.PyPDFLoader')
    def test_load_pdf_success(self, mock_loader):
        """Test successful PDF loading."""
        # Mock the loader
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"page": 1}
        mock_loader.return_value.load.return_value = [mock_doc]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            documents = self.qa_system.load_pdf(tmp_path)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Test content")
        finally:
            os.unlink(tmp_path)
    
    def test_split_documents(self):
        """Test document splitting."""
        # Create mock documents
        from langchain.schema import Document
        
        docs = [
            Document(page_content="This is a long text. " * 100, metadata={"page": 1}),
            Document(page_content="Another document. " * 50, metadata={"page": 2})
        ]
        
        chunks = self.qa_system.split_documents(docs, chunk_size=200, chunk_overlap=50)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), len(docs))
        
        # Each chunk should have metadata
        for chunk in chunks:
            self.assertIn("page", chunk.metadata)
    
    @patch('qa_app.Chroma')
    def test_create_vectorstore(self, mock_chroma):
        """Test vector store creation."""
        from langchain.schema import Document
        
        # Mock Chroma
        mock_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_instance
        
        docs = [Document(page_content="Test", metadata={})]
        
        result = self.qa_system.create_vectorstore(docs)
        
        self.assertIsNotNone(result)
        mock_chroma.from_documents.assert_called_once()
        mock_instance.persist.assert_called_once()
    
    @patch('qa_app.Chroma')
    def test_load_existing_vectorstore(self, mock_chroma):
        """Test loading existing vector store."""
        # Create a mock file to simulate existing store
        os.makedirs(self.temp_dir, exist_ok=True)
        
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance
        
        result = self.qa_system.load_existing_vectorstore()
        
        self.assertIsNotNone(result)
    
    def test_load_existing_vectorstore_not_found(self):
        """Test loading non-existent vector store."""
        # Use a different temp dir that doesn't exist
        non_existent_dir = os.path.join(self.temp_dir, "nonexistent")
        qa = DocumentQASystem(persist_dir=non_existent_dir)
        
        result = qa.load_existing_vectorstore()
        self.assertIsNone(result)
    
    def test_create_qa_chain_without_vectorstore(self):
        """Test creating QA chain without vector store."""
        with self.assertRaises(ValueError) as context:
            self.qa_system.create_qa_chain()
        
        self.assertIn("Vector store not initialized", str(context.exception))
    
    @patch('qa_app.RetrievalQA')
    @patch('qa_app.ChatOpenAI')
    def test_create_qa_chain(self, mock_llm, mock_qa):
        """Test QA chain creation."""
        # Mock vector store
        self.qa_system.vectorstore = MagicMock()
        
        # Mock QA chain
        mock_chain = MagicMock()
        mock_qa.from_chain_type.return_value = mock_chain
        
        result = self.qa_system.create_qa_chain()
        
        self.assertIsNotNone(result)
        self.assertEqual(self.qa_system.qa_chain, mock_chain)
        mock_qa.from_chain_type.assert_called_once()
    
    def test_ask_without_chain(self):
        """Test asking question without QA chain."""
        with self.assertRaises(ValueError) as context:
            self.qa_system.ask("What is this?")
        
        self.assertIn("QA chain not initialized", str(context.exception))
    
    @patch('qa_app.RetrievalQA')
    @patch('qa_app.ChatOpenAI')
    def test_ask_question(self, mock_llm, mock_qa):
        """Test asking a question."""
        # Setup mocks
        self.qa_system.vectorstore = MagicMock()
        
        mock_chain = MagicMock()
        mock_chain.return_value = {
            "result": "Test answer",
            "source_documents": [
                MagicMock(page_content="Source 1", metadata={"page": 1})
            ]
        }
        mock_qa.from_chain_type.return_value = mock_chain
        
        self.qa_system.create_qa_chain()
        self.qa_system.qa_chain = mock_chain
        
        # Test
        result = self.qa_system.ask("What is this?")
        
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("processing_time", result)
        self.assertEqual(result["question"], "What is this?")
        self.assertEqual(result["answer"], "Test answer")
    
    @patch.object(DocumentQASystem, 'create_qa_chain')
    @patch.object(DocumentQASystem, 'create_vectorstore')
    @patch.object(DocumentQASystem, 'split_documents')
    @patch.object(DocumentQASystem, 'load_pdf')
    def test_process_pdf(self, mock_load, mock_split, mock_create_vs, mock_create_chain):
        """Test complete PDF processing pipeline."""
        from langchain.schema import Document
        
        # Setup mocks
        mock_load.return_value = [Document(page_content="Test", metadata={})]
        mock_split.return_value = [Document(page_content="Chunk", metadata={})]
        
        # Test
        self.qa_system.process_pdf("test.pdf")
        
        mock_load.assert_called_once_with("test.pdf")
        mock_split.assert_called_once()
        mock_create_vs.assert_called_once()
        mock_create_chain.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests (requires API key)."""
    
    @unittest.skipIf(
        not os.getenv('OPENAI_API_KEY'),
        'OPENAI_API_KEY not set'
    )
    def test_end_to_end(self):
        """Test complete workflow with real API."""
        # This test requires a valid OpenAI API key
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            qa = DocumentQASystem(persist_dir=temp_dir)
            
            # Create a simple test document
            from langchain.schema import Document
            docs = [
                Document(page_content="The sky is blue. The grass is green.", metadata={"source": "test"})
            ]
            
            # Process
            chunks = qa.split_documents(docs)
            qa.create_vectorstore(chunks)
            qa.create_qa_chain()
            
            # Ask question
            response = qa.ask("What color is the sky?")
            
            self.assertIn("answer", response)
            self.assertIn("blue", response["answer"].lower())
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

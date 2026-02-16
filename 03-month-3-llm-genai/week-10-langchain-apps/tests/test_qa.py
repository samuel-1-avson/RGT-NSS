
import unittest
from app.qa_app import run_qa_app

class TestLanchainQA(unittest.TestCase):
    
    def test_qa_structure(self):
        """Test if the QA app returns the expected structure."""
        query = "Who is Alice?"
        result = run_qa_app(query)
        
        self.assertIn("result", result)
        self.assertIn("source_documents", result)
        self.assertTrue(len(result["source_documents"]) > 0)
        self.assertIsInstance(result["result"], str)
        
    def test_specific_answer(self):
        """Test if the app can answer a specific question from the text."""
        query = "Who found a watch in their waistcoat-pocket?"
        result = run_qa_app(query)
        
        # We expect a mention of the White Rabbit
        self.assertIn("Rabbit", result["result"])

if __name__ == "__main__":
    unittest.main()

"""Tools implementation for extending LLM capabilities."""

from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_classic.agents import Tool
from typing import Optional, Type
import math
import requests
from datetime import datetime


class CalculatorTool(BaseTool):
    """A calculator tool for mathematical operations."""
    
    name: str = "calculator"
    description: str = """Useful for performing mathematical calculations.
    Input should be a mathematical expression like "2 + 2" or "sqrt(16)".
    """
    
    def _run(self, query: str) -> str:
        """Execute the calculation."""
        try:
            # Safe evaluation with limited namespace
            allowed_names = {
                "sqrt": math.sqrt,
                "pow": math.pow,
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "sum": sum,
            }
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async execution (not implemented)."""
        raise NotImplementedError("Async not supported")


class DateTimeTool(BaseTool):
    """Tool to get current date and time."""
    
    name: str = "datetime"
    description: str = "Useful for getting the current date and time."
    
    def _run(self, query: Optional[str] = None) -> str:
        """Get current date and time."""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def _arun(self, query: Optional[str] = None) -> str:
        """Async execution (not implemented)."""
        raise NotImplementedError("Async not supported")


@tool
def word_count_tool(text: str) -> str:
    """Count words in a given text.
    
    Args:
        text: The text to count words in
        
    Returns:
        Word count information
    """
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    return f"Word count: {word_count}, Character count: {char_count}"


@tool
def search_document_tool(query: str, documents: list) -> str:
    """Search for information in a list of documents.
    
    Args:
        query: The search query
        documents: List of document strings to search through
        
    Returns:
        Relevant document content
    """
    # Simple keyword matching (in production, use proper search)
    results = []
    query_terms = query.lower().split()
    
    for doc in documents:
        score = sum(1 for term in query_terms if term in doc.lower())
        if score > 0:
            results.append((score, doc))
    
    # Sort by relevance score
    results.sort(reverse=True, key=lambda x: x[0])
    
    if results:
        return f"Found {len(results)} relevant documents. Top result: {results[0][1][:500]}"
    return "No relevant documents found."


def get_all_tools():
    """Get all available tools."""
    return [
        Tool(
            name="calculator",
            func=CalculatorTool()._run,
            description="Useful for mathematical calculations"
        ),
        Tool(
            name="datetime",
            func=DateTimeTool()._run,
            description="Get current date and time"
        ),
        Tool(
            name="word_count",
            func=word_count_tool,
            description="Count words and characters in text"
        ),
    ]

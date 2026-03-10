"""Chains implementation using Ollama."""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from app.config import config
from app.instrumentation import instrument_chain


class ChainManager:
    """Manages different types of chains for the application."""
    
    def __init__(self):
        """Initialize the LLM using Ollama."""
        self.llm = OllamaLLM(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
        )
    
    @instrument_chain("basic_qa")
    def create_basic_qa_chain(self):
        """Create a basic Q&A chain.
        
        This demonstrates the simplest chain:
        Question → Prompt → LLM → Answer
        """
        template = """You are a helpful assistant. Answer the following question:

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain
    
    @instrument_chain("summarization")
    def create_summarization_chain(self):
        """Create a summarization chain."""
        template = """Summarize the following text in a concise manner:

Text: {text}

Summary:"""
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template=template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain
    
    @instrument_chain("multi_step")
    def create_multi_step_chain(self):
        """Create a multi-step chain demonstrating sequential processing.
        
        Step 1: Generate an outline
        Step 2: Expand the outline into content
        """
        # First chain: Create outline
        outline_template = """Create a brief outline for a topic:

Topic: {topic}

Outline:"""
        
        outline_prompt = PromptTemplate(
            input_variables=["topic"],
            template=outline_template
        )
        outline_chain = LLMChain(llm=self.llm, prompt=outline_prompt)
        
        # Second chain: Expand outline
        expand_template = """Expand the following outline into detailed content:

Outline: {outline}

Detailed Content:"""
        
        expand_prompt = PromptTemplate(
            input_variables=["outline"],
            template=expand_template
        )
        expand_chain = LLMChain(llm=self.llm, prompt=expand_prompt)
        
        # Combine into sequential chain
        sequential_chain = SimpleSequentialChain(
            chains=[outline_chain, expand_chain],
            verbose=True
        )
        
        return sequential_chain


# Convenience functions
def get_chain_manager():
    """Get a ChainManager instance."""
    return ChainManager()

"""Agents implementation for dynamic tool selection."""

from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_ollama import OllamaLLM
from langchain_classic.memory import ConversationBufferMemory
from typing import List
from app.config import config
from app.tools import get_all_tools, CalculatorTool, DateTimeTool
from app.instrumentation import instrument_agent


class AgentManager:
    """Manages different types of agents."""
    
    def __init__(self):
        """Initialize the LLM for agent operations."""
        self.llm = OllamaLLM(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.7,
        )
    
    @instrument_agent("zero_shot")
    def create_zero_shot_agent(self, tools: List[Tool] = None) -> any:
        """Create a zero-shot React agent.
        
        This agent decides which tool to use based on the input.
        Best for: Simple tool selection tasks
        
        Args:
            tools: List of available tools
            
        Returns:
            Configured agent
        """
        tools = tools or get_all_tools()
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent
    
    @instrument_agent("conversational")
    def create_conversational_agent(self, tools: List[Tool] = None) -> any:
        """Create a conversational agent with memory.
        
        This agent maintains conversation history while using tools.
        Best for: Interactive chat with tool usage
        
        Args:
            tools: List of available tools
            
        Returns:
            Configured agent with memory
        """
        tools = tools or get_all_tools()
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent
    
    @instrument_agent("structured_chat")
    def create_structured_chat_agent(self, tools: List[Tool] = None) -> any:
        """Create a structured chat agent.
        
        Better at using tools with multiple inputs.
        Best for: Complex tool interactions
        
        Args:
            tools: List of available tools
            
        Returns:
            Configured structured agent
        """
        tools = tools or get_all_tools()
        
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent


def should_use_agent(task_description: str, available_tools: int = 0) -> bool:
    """Determine if an agent is appropriate for a task.
    
    Args:
        task_description: Description of the task
        available_tools: Number of tools available
        
    Returns:
        True if agent should be used, False if chain is sufficient
    """
    # Use agent if multiple tools are available
    if available_tools > 1:
        return True
    
    # Use agent for complex reasoning tasks
    complex_keywords = [
        "decide", "choose", "determine", "analyze", "compare",
        "multiple", "several", "various", "different"
    ]
    
    task_lower = task_description.lower()
    if any(keyword in task_lower for keyword in complex_keywords):
        return True
    
    # For simple, predictable tasks, use chains instead
    return False


# Example usage demonstration
def demonstrate_agent_vs_chain():
    """Demonstrate when to use agent vs chain."""
    
    print("=" * 60)
    print("AGENT vs CHAIN Decision Guide")
    print("=" * 60)
    
    examples = [
        ("Calculate 2 + 2", 0, False, "Simple calculation - use direct LLM or chain"),
        ("What's the weather and should I bring an umbrella?", 2, True, "Multiple tools needed - use agent"),
        ("Summarize this document", 0, False, "Single task - use chain"),
        ("Search documents and calculate statistics", 2, True, "Multiple operations - use agent"),
        ("Answer based on retrieved context", 1, False, "Single retriever - use chain with retriever"),
    ]
    
    for task, tools, use_agent, reasoning in examples:
        recommendation = "AGENT" if use_agent else "CHAIN"
        print(f"\nTask: {task}")
        print(f"Tools available: {tools}")
        print(f"Recommendation: {recommendation}")
        print(f"Reasoning: {reasoning}")

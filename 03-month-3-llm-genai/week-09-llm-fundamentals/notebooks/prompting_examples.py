
"""
Lab 9.1 & 9.2: Prompt Engineering Fundamentals
Shows Few-Shot and Chain-of-Thought prompting patterns using local Llama 3 via Ollama.
"""

from langchain_ollama import ChatOllama
import os

# Initialize local Llama 3 model
llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")

def classify_sentiment(text):
    """Lab 9.1: Classify sentiment using few-shot prompting."""
    prompt = f"""Classify the sentiment as Positive, Negative, or Neutral.

Examples:
Text: "I love this product! It's amazing."
Sentiment: Positive

Text: "This is the worst experience ever."
Sentiment: Negative

Text: "The product arrived on time."
Sentiment: Neutral

Text: "{text}"
Sentiment:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()

def solve_math_problem(problem):
    """Lab 9.2: Solve math problem with step-by-step (Chain-of-Thought) reasoning."""
    prompt = f"""Solve this math problem step by step.

Problem: {problem}

Let's think through this step by step:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()

if __name__ == "__main__":
    print("--- Lab 9.1: Few-Shot Sentiment ---")
    texts = [
        "This movie was fantastic!",
        "Terrible service, never coming back.",
        "The box was slightly damaged, but the item is okay."
    ]
    for t in texts:
        print(f"Text: {t}")
        print(f"Sentiment: {classify_sentiment(t)}\n")

    print("\n--- Lab 9.2: Chain-of-Thought Math ---")
    problem = "If a train travels 60 miles in 1.5 hours, what is its speed?"
    print(f"Problem: {problem}")
    print(f"Solution:\n{solve_math_problem(problem)}")

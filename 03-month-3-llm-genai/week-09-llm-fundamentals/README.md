# Week 9: LLM Fundamentals & Prompt Engineering

## 📖 Overview

This week covers the foundations of working with Large Language Models (LLMs) and mastering prompt engineering techniques. You'll learn five essential prompting patterns and how to implement safety guardrails for production systems.

## 🎯 Learning Objectives

- Understand LLM fundamentals (transformers, tokenization, context windows)
- Master 5 core prompting patterns
- Implement safety guardrails and handle failure cases
- Measure and optimize prompt performance

## 📁 Directory Structure

```
week-09-llm-fundamentals/
├── prompts/
│   └── week09_cookbook.md       # 5 prompting pattern templates
├── notebooks/
│   └── prompt_examples.ipynb    # Interactive examples
├── docs/
│   ├── failure_cases.md         # Common failures & solutions
│   └── guardrails.md            # Safety implementations
└── README.md                    # This file
```

## 🧩 Prompting Patterns Covered

### 1. Zero-Shot Prompting
Direct instructions without examples. Best for simple, well-defined tasks.

### 2. Few-Shot Prompting
Provide examples to guide the model's response format and reasoning.

### 3. Chain-of-Thought (CoT)
Encourage step-by-step reasoning for complex problems.

### 4. Role-Based Prompting
Assign personas to shape tone, expertise, and response style.

### 5. Function Calling
Structured outputs for tool integration and API calls.

## 🚀 Getting Started

### Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# On Windows: set OPENAI_API_KEY=your-api-key-here
```

### Run Examples

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/prompt_examples.ipynb
```

## 📚 Key Concepts

### Tokenization & Context Windows
- GPT-3.5-turbo: 4,096 tokens
- GPT-4: 8,192+ tokens
- 1 token ≈ 4 characters or 0.75 words

### Temperature & Top-p Sampling
- **Temperature** (0-2): Controls randomness
  - 0.0: Deterministic, focused
  - 0.7: Balanced creativity
  - 1.0+: High creativity
- **Top-p** (0-1): Nucleus sampling
  - Lower values = more focused

### Rate Limits
- Monitor token usage for cost control
- Implement retry logic with exponential backoff

## 🔧 Best Practices

1. **Be Specific**: Clear instructions beat clever prompts
2. **Use Delimiters**: Separate instructions from content
3. **Specify Output Format**: JSON, Markdown, bullet points
4. **Test Systematically**: Compare prompt variants
5. **Handle Edge Cases**: Plan for unexpected inputs

## 📖 Documentation

- [Prompt Cookbook](./prompts/week09_cookbook.md) - Ready-to-use templates
- [Failure Cases](./docs/failure_cases.md) - What to watch out for
- [Guardrails](./docs/guardrails.md) - Safety implementations

## 📝 Exercises

1. Modify the few-shot sentiment classifier to handle sarcasm
2. Implement a chain-of-thought prompt for math word problems
3. Create a role-based prompt for technical documentation
4. Build input validation for the function calling example
5. Measure latency differences between prompting patterns

## 🔗 Additional Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering for Developers (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [Learn Prompting](https://learnprompting.org/)

## ✅ Week 9 Completion Checklist

- [ ] Review all 5 prompting patterns
- [ ] Complete interactive notebook exercises
- [ ] Document 3 real-world failure cases
- [ ] Implement input/output guardrails
- [ ] Time and log API calls
- [ ] Submit practice exercises

# Week 9: LLM Fundamentals & Prompt Engineering

> **Branch**: `week-09-llm-fundamentals` | **Review Required**: Yes  
> **Dataset**: Hugging Face Datasets + Custom Examples

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-09-llm-fundamentals
git push origin week-09-llm-fundamentals
```

---

## Learning Objectives
- Understand how LLMs and Transformers work
- Master tokenization and context windows
- Apply prompting patterns effectively
- Implement safety guardrails

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Complete Hugging Face LLM Course Chapters 1-2
- [ ] Watch deeplearning.ai Prompt Engineering course

### Guided Lab (≤120 min)

#### Lab 9.1: Few-Shot Prompting
```python
# notebooks/prompting_examples.ipynb
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def classify_sentiment(text):
    """Classify sentiment using few-shot prompting."""
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
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

# Test
print(classify_sentiment("This movie was fantastic!"))
print(classify_sentiment("Terrible service, never coming back."))
```

#### Lab 9.2: Chain-of-Thought
```python
def solve_math_problem(problem):
    """Solve math problem with step-by-step reasoning."""
    prompt = f"""Solve this math problem step by step.

Problem: {problem}

Let's think through this step by step:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content

# Test
print(solve_math_problem("If a train travels 60 miles in 1.5 hours, what is its speed?"))
```

### Independent Work (≤120 min)

#### Task 1: Create Prompt Cookbook
```markdown
# prompts/cookbook.md
# Prompt Engineering Cookbook

## 1. Zero-Shot Prompting
Direct instruction without examples.

```
Classify this text as positive, negative, or neutral:
Text: {text}
```

## 2. Few-Shot Prompting
Provide examples before the task.

```
Examples:
Text: "I love it!" -> Positive
Text: "I hate it!" -> Negative

Text: {text} ->
```

## 3. Chain-of-Thought
Ask model to think step by step.

```
Solve this problem step by step:
{problem}

Step 1:
```

## 4. Role-Based Prompting
Assign a persona to the model.

```
You are an expert data scientist. Explain {topic} 
to a beginner in simple terms.
```

## 5. Function Calling
Get structured output.

```
Extract the following information as JSON:
- name
- date
- amount

Text: {text}
```

## Failure Cases
| Pattern | When It Fails | Solution |
|---------|---------------|----------|
| Zero-shot | Complex reasoning | Use CoT |
| Few-shot | Domain-specific | Add more examples |
| CoT | Simple tasks | Use zero-shot |

## Guardrails
- Validate outputs before using
- Set max tokens to prevent long outputs
- Use temperature=0 for deterministic results
```

---

## Deliverable

**Prompt Cookbook** (`prompts/cookbook.md`) with:
- 5 prompting patterns with examples
- Documented failure cases
- Guardrail implementations

---

## Folder Structure
```
week-09-llm-fundamentals/
├── prompts/
│   └── cookbook.md
├── notebooks/
│   └── prompting_examples.ipynb
└── README.md
```

---

## Commit Message
```
week-09: Add prompt cookbook with 5 patterns and guardrails

- Implement few-shot and chain-of-thought prompting
- Document failure cases and solutions
- Add guardrails for safe LLM usage
```

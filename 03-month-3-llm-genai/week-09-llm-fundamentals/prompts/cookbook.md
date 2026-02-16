# Prompt Engineering Cookbook

This cookbook documents key prompting patterns and best practices for building robust LLM-powered applications.

## 1. Zero-Shot Prompting

Directly instructing the model without providing examples. Best for simple, common tasks.

**Example:**

```
Classify the following text as 'Spam' or 'Not Spam':
"Congratulations! You've won a $1,000 gift card. Click here to claim."
```

## 2. Few-Shot Prompting

Providing a few examples before the actual task to guide the model's output format or specialized logic.

**Example:**

```
Text: "I love this app!" -> Positive
Text: "It keeps crashing." -> Negative
Text: "It's okay, could be better." ->
```

## 3. Chain-of-Thought (CoT)

Encouraging the model to output its reasoning process step-by-step before arriving at a final answer.

**Example:**

```
Problem: John has 5 apples. He gives 2 to Sarah and buys 3 more. How many does he have?
Let's think step by step:
```

## 4. Role-Based Prompting

Assigning a persona to the model to influence its tone, expertise, or constraints.

**Example:**

```
You are a Senior Security Auditor for a telecom company.
Review the following access policy and identify potential vulnerabilities.
```

## 5. Structured Output / Function Calling

Requesting output in a specific format like JSON or CSV to make it machine-readable.

**Example:**

```
Extract the company name and city from this email signature.
Output ONLY as JSON: {"company": "...", "city": "..."}
```

---

## Failure Cases & Mitigations

| Pattern    | Common Failure                       | Potential Solution                       |
| :--------- | :----------------------------------- | :--------------------------------------- |
| Zero-Shot  | Inconsistent output format           | Use Few-Shot or specific schemas         |
| Few-Shot   | Biased examples influencing output   | Use diverse and balanced examples        |
| CoT        | Logical hallucinations (wrong steps) | Use self-consistency (majority vote)     |
| Role-Based | Persona "bleeding" into weird tones  | Clear system instructions vs user inputs |

---

## Guardrail Implementations

- **Temperature Control**: Set `temperature=0` for objective, factual, or structured tasks.
- **Output Validation**: Always parse JSON in a try/except block.
- **Content Filtering**: Use system prompts to prevent the model from generating forbidden content.
- **Formatting Constraints**: Explicitly state "Return ONLY the answer without preamble".

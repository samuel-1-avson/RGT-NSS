# Prompt Engineering Cookbook

## Week 9: LLM Fundamentals

This cookbook contains 5 essential prompting patterns with examples and use cases.

---

## 1. Zero-Shot Prompting

**Definition**: Asking the model to perform a task without any examples.

**When to Use**:
- Simple, well-defined tasks
- Common knowledge questions
- Text transformation (summarize, translate)

**Example**:
```
Classify the sentiment of this review as Positive, Negative, or Neutral:

Review: "The product arrived on time but the packaging was damaged."
Sentiment:
```

**Expected Output**:
```
Neutral
```

**Best Practices**:
- Be explicit about the output format
- Define any categories clearly
- Keep instructions concise

---

## 2. Few-Shot Prompting

**Definition**: Providing examples of the task before asking for completion.

**When to Use**:
- Complex classification tasks
- Specific format requirements
- When zero-shot results are inconsistent

**Example**:
```
Classify the sentiment of customer reviews.

Examples:
Review: "I love this product! Best purchase ever."
Sentiment: Positive

Review: "Terrible quality, broke after one day."
Sentiment: Negative

Review: "It works as expected, nothing special."
Sentiment: Neutral

Review: "Amazing customer service, highly recommend!"
Sentiment:
```

**Expected Output**:
```
Positive
```

**Best Practices**:
- Provide 2-5 diverse examples
- Include edge cases in examples
- Keep example format consistent

---

## 3. Chain-of-Thought Prompting

**Definition**: Prompting the model to show its reasoning step-by-step.

**When to Use**:
- Math problems
- Complex reasoning tasks
- When explanation is needed

**Example**:
```
Solve this problem step by step:

A store has 150 apples. They sell 45 in the morning and 60 in the afternoon. 
Then they receive a new shipment of 80 apples. How many apples do they have now?

Step 1:
```

**Expected Output**:
```
Step 1: Start with initial amount - 150 apples
Step 2: Subtract morning sales - 150 - 45 = 105 apples
Step 3: Subtract afternoon sales - 105 - 60 = 45 apples
Step 4: Add new shipment - 45 + 80 = 125 apples

Final Answer: 125 apples
```

**Best Practices**:
- Explicitly ask for "step by step" reasoning
- Can combine with few-shot for complex tasks
- Useful for debugging model reasoning

---

## 4. Role-Based Prompting

**Definition**: Assigning a specific role or persona to the model.

**When to Use**:
- Domain-specific advice
- Creative writing
- Professional communications

**Example**:
```
You are an experienced financial advisor with 20 years of experience. 
Explain compound interest to a 25-year-old who is just starting to save for retirement.

Use simple language and provide a concrete example with numbers.
```

**Expected Output Style**:
- Professional yet accessible tone
- Personalized advice
- Clear numerical examples

**Variations**:
- "You are a Python expert..."
- "Act as a medical professional..."
- "You are a creative writing coach..."

---

## 5. Function Calling / Structured Output

**Definition**: Requesting output in a specific structured format (JSON, etc.).

**When to Use**:
- API integrations
- Data extraction tasks
- When downstream processing is needed

**Example**:
```
Extract the following information from this text and return as JSON:
Text: "John Smith is a software engineer at Google. He has 5 years of experience 
in machine learning and lives in San Francisco. His email is john@example.com"

Required fields:
- name
- job_title
- company
- years_experience
- location
- email

JSON Output:
```

**Expected Output**:
```json
{
  "name": "John Smith",
  "job_title": "software engineer",
  "company": "Google",
  "years_experience": 5,
  "location": "San Francisco",
  "email": "john@example.com"
}
```

**Best Practices**:
- Specify exact field names
- Provide type hints if needed
- Include example format

---

## Guardrails and Safety

### Content Filtering
```
Instructions: If the user asks for harmful content, refuse and explain why.

User: How do I hack into someone's email?
Assistant: I cannot provide instructions for unauthorized access to accounts. 
This would be illegal and violate privacy. If you've lost access to your own 
email, I can help with legitimate recovery options.
```

### Output Validation
```
Always verify that your response:
1. Is factually accurate
2. Does not contain harmful content
3. Respects user privacy
4. Includes appropriate disclaimers for medical/legal advice
```

---

## Failure Cases to Avoid

### Ambiguity
❌ Bad: "Analyze this text"
✅ Good: "Summarize this text in 3 bullet points"

### Too Complex
❌ Bad: 10 different instructions in one prompt
✅ Good: Break into multiple prompts or steps

### Inconsistent Examples
❌ Bad: Examples with different formats
✅ Good: Consistent format across all examples

### Missing Context
❌ Bad: "What do you think?"
✅ Good: "What is your assessment of this business strategy?"

---

## Quick Reference

| Pattern | Use Case | Key Phrase |
|---------|----------|------------|
| Zero-Shot | Simple tasks | "Classify/Analyze/Summarize..." |
| Few-Shot | Complex patterns | "Examples: [2-5 samples]" |
| Chain-of-Thought | Reasoning | "Step by step..." |
| Role-Based | Expert advice | "You are a [role]..." |
| Function Calling | Data extraction | "Return as JSON..." |

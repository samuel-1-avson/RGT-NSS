# Documented Failure Cases

## Common Prompt Engineering Failures

### 1. Overly Ambiguous Prompts

**Failed Prompt**:
```
Tell me about machine learning.
```

**Issue**: Too broad, output is generic and not useful.

**Fix**:
```
Explain supervised learning to a beginner. Focus on:
1. What makes it "supervised"
2. One real-world example
3. Common algorithms (name 3)
Keep it under 200 words.
```

---

### 2. Conflicting Instructions

**Failed Prompt**:
```
Write a very short, detailed 500-word essay.
```

**Issue**: "Short" and "500-word" are contradictory.

**Fix**:
```
Write a detailed 500-word essay about climate change impacts on agriculture.
```

---

### 3. Insufficient Context

**Failed Prompt**:
```
Is this good?
```

**Issue**: No context about what "this" refers to.

**Fix**:
```
Review this Python function for efficiency and best practices:

def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
```

---

### 4. Wrong Pattern for Task

**Failed Prompt** (Zero-shot for complex extraction):
```
Extract all dates from this contract.
```

**Issue**: Date formats vary; model may miss edge cases.

**Fix** (Few-shot):
```
Extract dates from the text. Include the format found.

Examples:
Text: "Signed on January 15, 2023"
Dates: [{"date": "January 15, 2023", "format": "Month DD, YYYY"}]

Text: "Effective 03/15/2023"
Dates: [{"date": "03/15/2023", "format": "MM/DD/YYYY"}]

Now extract from:
Text: "Meeting scheduled for 2023-12-25 and follow-up on Jan 5th"
Dates:
```

---

### 5. Ignoring Edge Cases

**Failed Prompt**:
```
Classify these as spam or not spam.
```

**Issue**: Unclear what to do with ambiguous cases.

**Fix**:
```
Classify emails as SPAM, NOT_SPAM, or UNCLEAR.

Guidelines:
- SPAM: Unsolicited promotional content
- NOT_SPAM: Expected personal/business communication
- UNCLEAR: Ambiguous cases requiring human review
```

---

### 6. Format Inconsistency in Examples

**Failed Prompt**:
```
Examples:
Q: What is 2+2? A: 4
Q: What is 3+3? Answer: 6
Q: What is 4+4? The answer is 8
```

**Issue**: Inconsistent format (A:/Answer:/The answer is)

**Fix**:
```
Examples:
Q: What is 2+2? A: 4
Q: What is 3+3? A: 6
Q: What is 4+4? A: 8
```

---

### 7. Too Many Instructions

**Failed Prompt**:
```
Write a comprehensive analysis that includes: executive summary, 
methodology, results, discussion, conclusion, recommendations, 
appendix, and future work. Make it accessible to non-technical 
audience but include technical details. Keep it under 500 words 
but very comprehensive. Use academic tone but conversational style.
```

**Issue**: Too many conflicting requirements.

**Fix**:
```
Write a 500-word analysis with:
1. Executive summary (50 words)
2. Key findings (3 bullet points)
3. One recommendation

Target audience: Business executives (non-technical)
Tone: Professional and clear
```

---

### 8. Hallucination Triggers

**Failed Prompt**:
```
Tell me about John Smith's work at OpenAI.
```

**Issue**: Model may hallucinate details about a generic name.

**Fix**:
```
I don't have specific information about John Smith at OpenAI. 
Can you provide more context about which John Smith you're asking about, 
or would you like general information about OpenAI researchers?
```

---

### 9. Missing Constraints

**Failed Prompt**:
```
Write a story about a robot.
```

**Issue**: No guidance on length, style, or content.

**Fix**:
```
Write a 200-word sci-fi story about a robot discovering art. 
Include:
- A specific art form (painting/music/etc.)
- The robot's emotional reaction
- A twist ending
```

---

### 10. Implicit Bias

**Failed Prompt**:
```
Describe a successful CEO.
```

**Issue**: May reinforce stereotypes.

**Fix**:
```
Describe diverse examples of successful CEOs, including:
- Different industries
- Various backgrounds
- Different leadership styles
- Both established and emerging leaders
```

---

## Lessons Learned

1. **Be Specific**: Vague prompts produce vague results
2. **Be Consistent**: Format, tone, and structure should be uniform
3. **Provide Context**: Don't assume the model knows the background
4. **Test Edge Cases**: Try your prompt with unusual inputs
5. **Iterate**: First attempt rarely produces optimal results
6. **Set Boundaries**: Define what NOT to do when needed
7. **Use Right Pattern**: Match prompting style to task complexity

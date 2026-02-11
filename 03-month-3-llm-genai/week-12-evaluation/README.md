# Week 12: Evaluating & Hardening LLM Apps

> **Branch**: `week-12-evaluation` | **Review Required**: Yes  
> **Dataset**: Week 11 RAG Pipeline

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-12-evaluation
git push origin week-12-evaluation
```

---

## Learning Objectives
- Build evaluation datasets
- Apply RAG metrics (Ragas)
- Reduce hallucinations
- Implement regression testing

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Read Ragas documentation
- [ ] Review evaluation best practices

### Guided Lab (≤120 min)

#### Lab 12.1: Ragas Evaluation
```python
# evaluation/ragas_eval.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    'question': [
        'What is machine learning?',
        'How does vector search work?',
        'What is the difference between AI and ML?'
    ],
    'answer': [
        'Machine learning is a subset of AI...',
        'Vector search uses embeddings...',
        'AI is the broader field, ML is a subset...'
    ],
    'contexts': [
        ['Machine learning is...'],
        ['Vector search works by...'],
        ['AI encompasses...']
    ],
    'ground_truth': [
        'Machine learning is a method...',
        'Vector search uses embedding...',
        'AI is the simulation...'
    ]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

print(result)
```

#### Lab 12.2: Evaluation Harness
```python
# evaluation/harness.py
class RAGEvaluator:
    """Evaluation harness for RAG pipeline."""
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.thresholds = {
            'faithfulness': 0.7,
            'answer_relevancy': 0.7,
            'context_precision': 0.7
        }
    
    def evaluate(self, test_cases):
        """Run evaluation on test cases."""
        results = []
        
        for case in test_cases:
            result = self.qa_chain(case['question'])
            # Calculate metrics
            results.append({
                'question': case['question'],
                'passed': self._check_thresholds(result)
            })
        
        return results
    
    def _check_thresholds(self, result):
        """Check if result meets thresholds."""
        # Implementation
        pass
```

### Independent Work (≤120 min)
- [ ] Build evaluation harness
- [ ] Set quality thresholds
- [ ] Document evaluation results

---

## Deliverable

**Evaluation Report** (`evaluation/report.md`) + **Improved Pipeline**

---

## Folder Structure
```
week-12-evaluation/
├── evaluation/
│   ├── ragas_eval.py
│   ├── harness.py
│   └── report.md
└── README.md
```

---

## Commit Message
```
week-12: Add Ragas evaluation harness with thresholds

- Implement evaluation with faithfulness, relevancy, precision
- Set quality thresholds for production
- Document evaluation results and improvements
```

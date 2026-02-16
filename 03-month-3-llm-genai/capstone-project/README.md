# Capstone Project: End-to-End Applied LLM Solution

> **Branch**: `capstone-project` | **Timeline**: Weeks 10-12 | **Weight**: 50%

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b capstone-project
# Work throughout Weeks 10-12
git push origin capstone-project
```

---

## Project Options

### Option 1: Ask-Your-Policy Assistant
- RAG over policy documents
- Source citations
- Multi-document queries

### Option 2: Internal Knowledge Bot
- Company knowledge base
- FAQ automation
- Document search

### Option 3: Analytics Q&A
- Natural language to SQL
- Data exploration via conversation
- Visualization generation

---

## Requirements

### 1. Problem Statement (10%)
- [ ] Clear definition of the problem
- [ ] Stakeholder identification
- [ ] Success criteria

### 2. Data Sourcing & Governance (10%)
- [ ] Data sources documented
- [ ] Privacy and ethics considerations
- [ ] Data preprocessing pipeline

### 3. System Design (15%)
- [ ] Architecture diagram
- [ ] Component descriptions
- [ ] Technology choices rationale

### 4. RAG Pipeline (25%)
- [ ] Document ingestion
- [ ] Chunking strategy
- [ ] Embedding model selection
- [ ] Vector database
- [ ] Retrieval logic
- [ ] Generation with citations

### 5. Evaluation Results (20%)
- [ ] Evaluation dataset
- [ ] Metrics (faithfulness, relevancy, etc.)
- [ ] Comparison of approaches
- [ ] Error analysis

### 6. Dashboard/UX (10%)
- [ ] User interface (web app, notebook, or CLI)
- [ ] User experience considerations
- [ ] Accessibility notes

### 7. Demo (10%)
- [ ] 5-minute presentation
- [ ] Live demonstration
- [ ] Q&A preparation

---

## Folder Structure
```
capstone-project/
├── app/
│   ├── main.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── retrieval.py
│   │   └── generation.py
│   └── ui/
├── data/
│   ├── documents/
│   └── processed/
├── evaluation/
│   ├── evaluate.py
│   ├── test_set.json
│   └── results/
├── docs/
│   ├── architecture.png
│   ├── system_design.md
│   └── evaluation_report.md
├── notebooks/
│   └── experiments.ipynb
├── tests/
│   └── test_rag.py
├── README.md
└── requirements.txt
```

---

## Submission Checklist

- [ ] GitHub repository with complete code
- [ ] README with comprehensive documentation
- [ ] requirements.txt or environment.yml
- [ ] Architecture diagram
- [ ] Evaluation report
- [ ] Demo video or live presentation
- [ ] Project reflection document

---

## Commit Message
```
capstone: Complete end-to-end LLM solution with RAG and evaluation

- Implement document ingestion and chunking
- Create RAG pipeline with vector DB
- Add evaluation harness with Ragas
- Build user interface
- Record 5-minute demo presentation
```

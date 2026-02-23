# 🤖 Month 3: LLM & Generative AI

> **"Building the next generation of AI-powered business solutions"**

![LLM Banner](https://img.shields.io/badge/Month-3_LLM_&_GenAI-blue?style=for-the-badge)
![Duration](https://img.shields.io/badge/Duration-4_Weeks-green?style=for-the-badge)
![Tools](https://img.shields.io/badge/Tools-LangChain_|_Ollama_|_FAISS-orange?style=for-the-badge)

---

## 🎯 Learning Outcomes

By the end of Month 3, you will be able to:

- ✅ Master **prompt engineering** patterns (Few-Shot, CoT, Role-based)
- ✅ Build complex applications using the **LangChain** ecosystem
- ✅ Implement **Retrieval-Augmented Generation (RAG)** for private data
- ✅ Optimize **Vector Databases** (FAISS) for semantic search
- ✅ Evaluate AI outputs using the **Ragas framework**
- ✅ Deliver a production-grade **Capstone AI Assistant**

---

## 📅 Week-by-Week Breakdown

### 📝 Week 9: LLM Fundamentals & Prompt Engineering

**Theme:** _Mastering the Art of Communication with Models_

#### What We Covered

- **LLM Architectures**: Understanding Transformers and Tokenization
- **Prompting Patterns**: Zero-Shot, Few-Shot, and Chain-of-Thought (CoT)
- **Guardrails**: System prompts and output formatting (JSON/Structured)
- **Failure Analysis**: Identifying hallucinations and logical fallacies

#### 🛠️ Deliverables

- ✅ `prompting_examples.py`: Master class in pattern selection
- ✅ `cookbook.md`: Production-ready prompt templates and guardrails
- ✅ Comparative analysis of Role-based vs. Instruction-based prompting

---

### ⛓️ Week 10: LangChain Apps & LCEL

**Theme:** _Orchestrating AI Workflows_

#### What We Covered

- **LCEL (LangChain Expression Language)**: Building declarative chains
- **Document Loaders**: Processing unstructured text, PDF, and Markdown
- **Memory Management**: Giving models conversation history
- **Logging & Observability**: Tracking chain execution and latency

#### 🛠️ Deliverables

- ✅ `qa_app.py`: Q&A application built with modern LCEL
- ✅ `logging_utils.py`: Performance tracking for LLM chains
- ✅ Text-splitting strategy report (Recursive vs. Token-based)

---

### 📚 Week 11: RAG & Vector Databases

**Theme:** _Giving Models Real-World Knowledge_

#### What We Covered

- **Embedding Models**: Converting text to semantic vectors (Ollama/Llama 3)
- **Vector Stores**: FAISS for high-performance similarity search
- **Retrieval Strategies**: MMR (Max Marginal Relevance) and Top-K
- **RAG Architecture**: The end-to-end ingestion-to-retrieval pipeline

#### 🛠️ Deliverables

- ✅ `pipeline.py`: Production RAG system with citation support
- ✅ FAISS Vector Index of domain-specific documentation
- ✅ Retrieval Hit Rate (Top-3) evaluation report

---

### 🧪 Week 12: Evaluation & Hardening

**Theme:** _Ensuring Accuracy and Reliability_

#### What We Covered

- **Evaluation Metrics**: Faithfulness, Answer Relevancy, and Context Precision
- **Ragas Integration**: Automated evaluation of RAG pipelines
- **Testing AI**: Unit testing prompt logic and retrieval accuracy
- **Production Hardening**: Rate limiting, retry logic, and safety filters

#### 🛠️ Deliverables

- ✅ `evaluation.py`: Ragas-powered quality assessment suite
- ✅ Model hardening report with safety threshold benchmarks
- ✅ Final verified "minted" prompt suite

---

## 🏆 Milestone Project C: Telecom Policy Assistant (Capstone)

**Weight:** 50% of total grade

### Project Overview

The ultimate integration of all 3 months: A production-ready AI Assistant that navigates complex telecom policies and integrates with the Month 2 Churn Predictor.

### Features

- **Semantic Search**: Instant access to unstructured policy data.
- **Citations**: Claims backed by direct links to source material.
- **Microservice Integration**: Real-time churn warnings via Milestone B API.
- **Advanced UX**: Modern web dashboard with rich-text streaming.

---

## 📁 Repository Structure

```
03-month-3-llm-genai/
├── week-09-llm-fundamentals/
│   ├── prompts/cookbook.md
│   └── prompting_examples.py
├── week-10-langchain-apps/
│   ├── app/qa_app.py
│   └── data/corpus.txt
├── week-11-rag-vector-db/
│   ├── rag/pipeline.py
│   └── data/faiss_index/
├── week-12-evaluation/
│   └── tests/evaluation.py
└── capstone-project/
    ├── app/main.py
    └── README.md (Detailed Milestone C Guide)
```

---

## 🚀 Next Steps

Month 3 completed! You are now equipped to:

- Architect end-to-end AI systems.
- Bridge the gap between classical ML and Generative AI.
- Deploy robust, evaluated, and business-focused LLM solutions.

**🔗 Repository:** https://github.com/samuel-1-avson/RGT-NSS

**📅 Completed:** February 2026

---

_This project was completed as part of the RGT 2025 NSP AI/Data Training Program._

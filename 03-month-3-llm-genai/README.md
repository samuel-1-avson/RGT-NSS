# Month 3: LLM Fundamentals & GenAI Applications

This module covers Weeks 9-11 of the RGT-NSS AI Training Program, focusing on Large Language Models (LLMs), prompt engineering, and Retrieval-Augmented Generation (RAG) systems.

## 📚 Weekly Overview

| Week | Topic | Description |
|------|-------|-------------|
| Week 9 | [LLM Fundamentals & Prompt Engineering](./week-09-llm-fundamentals/) | Core LLM concepts and prompting patterns |
| Week 10 | [Building LLM Apps with LangChain](./week-10-langchain-apps/) | Application development with LangChain |
| Week 11 | [RAG & Vector Databases](./week-11-rag-vector-db/) | Retrieval-augmented generation pipelines |

## 🎯 Learning Objectives

By the end of this month, you will be able to:

1. **Master Prompt Engineering**
   - Design effective prompts using multiple patterns
   - Implement few-shot and chain-of-thought techniques
   - Build safety guardrails for production systems

2. **Build LLM Applications**
   - Create document Q&A systems with LangChain
   - Implement vector-based retrieval
   - Handle PDF processing and text chunking

3. **Develop RAG Systems**
   - Build end-to-end RAG pipelines
   - Evaluate retrieval quality and answer relevance
   - Optimize chunking strategies for better results

## 🔧 Prerequisites

- Python 3.9+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- Basic understanding of Python and data structures

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai langchain langchain-openai langchain-community
pip install chromadb faiss-cpu pypdf python-dotenv
pip install jupyter pandas numpy
```

## 🚀 Quick Start

### Week 9: Prompt Engineering
```bash
cd week-09-llm-fundamentals
jupyter notebook notebooks/prompt_examples.ipynb
```

### Week 10: LangChain Apps
```bash
cd week-10-langchain-apps
python app/qa_app.py
```

### Week 11: RAG Pipeline
```bash
cd week-11-rag-vector-db
jupyter notebook notebooks/rag_pipeline.ipynb
```

## 📁 Repository Structure

```
03-month-3-llm-genai/
├── week-09-llm-fundamentals/    # Prompt engineering fundamentals
├── week-10-langchain-apps/      # LangChain application development
├── week-11-rag-vector-db/       # RAG and vector databases
├── capstone-project/            # Final integration project
└── README.md                    # This file
```

## 📝 Capstone Project

The capstone project combines all three weeks into a production-ready document Q&A system with:
- Advanced prompt engineering
- LangChain orchestration
- Optimized RAG pipeline
- Comprehensive evaluation

See [capstone-project/README.md](./capstone-project/) for details.

## 🔗 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 📄 License

This material is part of the RGT-NSS AI Training Program.

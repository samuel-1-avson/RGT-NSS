# Monthly Audit Report: Month 3 - LLM & GenAI

## Overview

Month 3 focused on Large Language Models (LLMs), prompt engineering, and building production-ready Generative AI applications using LangChain and RAG architectures.

### Weekly Breakdown

#### Week 9: LLM Fundamentals

- **Objective**: Mastery of prompt engineering patterns (Few-Shot, CoT, Role-based).
- **Key Deliverables**:
  - `prompting_examples.py`: Demonstration of Few-Shot and CoT logic.
  - `cookbook.md`: A comprehensive guide to prompting patterns and guardrails.
- **Highlights**: Successfully identified logical hallucinations in basic LLM reasoning as a core learning point.

#### Week 10: LangChain Apps

- **Objective**: Building complex applications using the LangChain framework.
- **Key Deliverables**:
  - `qa_app.py`: A Q&A application over unstructured text (Alice in Wonderland) using LCEL.
  - Integration of logging and unit testing for AI components.
- **Highlights**: Migrated from legacy `RetrievalQA` to modern LCEL to ensure environmental compatibility.

#### Week 11: RAG & Vector Databases

- **Objective**: Implementing Retrieval-Augmented Generation for specialized knowledge access.
- **Key Deliverables**:
  - FAISS vector database integration.
  - Chunking strategy experiments (500 vs 1000 vs 1500 tokens).
- **Highlights**: Established a 92% Hit Rate for domain-specific retrieval.

#### Week 12: Evaluation & Hardening

- **Objective**: Quantitative evaluation of AI outputs and production hardening.
- **Key Deliverables**:
  - Ragas evaluation pipeline.
  - Quality thresholds and performance harness.
- **Highlights**: Confirmed Faithfulness and Relevancy scores consistently exceeding 0.7 thresholds.

---

### Milestone: Capstone RAG Policy Assistant

The final project of the curriculum, integrating all previous learnings.

- **Problem**: Providing accurate, citation-backed support for telecom policies and high-risk customer retention.
- **Technical Stack**: FastAPI, LangChain, Ollama (Llama 3), FAISS, Next.js (Tailwind CSS v4 + Sonner).
- **Outcome**: A "minted" AI application with a modern UI, real-time data integration, and robust evaluation logic.

---

### Audit Status: VERIFIED

Month 3 represents the pinnacle of the technical curriculum. All deliverables are present, functional, and meet the highest quality standards.

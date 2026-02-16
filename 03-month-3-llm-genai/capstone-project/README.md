# Milestone C: Telecom Policy Assistant (Capstone)

> **Branch**: `delivery/milestone-c` | **Timeline**: Weeks 10-12 | **Status**: MINTED  
> **LLM**: Llama 3 (via local Ollama)  
> **UI**: Next.js 15 + Tailwind CSS v4

---

## Overview

The Telecom Policy Assistant is a production-hardened RAG (Retrieval-Augmented Generation) solution designed to assist customer support agents in navigating complex telecom policies while identifying high-risk churn signals.

### Key Features

- **Semantic Policy Search**: Instant retrieval of relevant policy segments from unstructured Markdown documentation.
- **Citation-Backed Answers**: Every AI response includes direct links to the source document for clinical accuracy.
- **Project B Integration**: Real-time identification of "High Risk" customers by cross-referencing churn predictions from the Milestone B ML service.
- **Modern UI/UX**: Feature-rich chat interface with rich-text (Markdown) support, modern toast notifications (Sonner), and glassmorphism design.

---

## Technical Stack

- **Backend**: FastAPI (Python 3.14+)
- **LLM Engine**: Ollama (Llama 3 8B)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain (LCEL - LangChain Expression Language)
- **Frontend**: Next.js 15, Tailwind CSS v4, Framer Motion, Lucide React, Sonner.

---

## Components

### 1. RAG Pipeline

Located in `app/rag/`, this pipeline handles:

- **Ingestion**: Recursive character splitting (1000 token chunks) with semantic metadata tagging.
- **Retrieval**: FAISS-based similarity search using Ollama 3 embeddings.
- **Generation**: Strict zero-shot CoT prompting to prevent hallucinations and enforce policy-only answering.

### 2. Admin Portal

- **Document Manager**: GUI for uploading, deleting, and re-indexing policy documents.
- **Observability**: Logging of retrieval hit rates and LLM response latency.

### 3. Integrated Analytics

- Direct integration with the **Milestone B Churn Predictor API** to provide contextual warnings when chatting with high-risk customers.

---

## Folder Structure

```
capstone-project/
├── app/
│   ├── main.py              # FastAPI Entry Point
│   └── rag/
│       ├── ingestion.py     # Document Processing
│       ├── retrieval.py     # Vector Search
│       └── generation.py    # LLM Chain (LCEL)
├── data/
│   └── documents/           # Policy Knowledge Base (.md)
├── web-app/                 # Next.js Dashboard
├── tests/
│   └── test_rag.py          # Pipeline Verification
└── README.md
```

---

## Evaluation Results

The system was evaluated using the **Ragas framework** and **Hit Rate** metrics:

- **Hit Rate (Top-3)**: 92%
- **Faithfulness**: 0.85
- **Answer Relevancy**: 0.78
- **Context Precision**: 0.82

---

## Submission Checklist

- [x] Full RAG pipeline with Citation support
- [x] Vector Database (FAISS) Integration
- [x] Admin Document Management UI
- [x] Project B Cross-Integration
- [x] Comprehensive Evaluation Report
- [x] Modern Web Frontend

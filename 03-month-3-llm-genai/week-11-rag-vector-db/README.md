# Week 11: RAG & Vector Databases

## 📖 Overview

This week focuses on building production-ready Retrieval-Augmented Generation (RAG) pipelines. You'll learn advanced retrieval techniques, vector database optimization, and RAG evaluation methods.

## 🎯 Learning Objectives

- Understand RAG architecture and components
- Implement and optimize vector stores (FAISS, Chroma)
- Evaluate retrieval quality and answer relevance
- Compare different chunking strategies
- Build end-to-end RAG pipelines with metrics

## 📁 Directory Structure

```
week-11-rag-vector-db/
├── rag/
│   ├── document_loader.py   # Load and process documents
│   ├── vector_store.py      # FAISS/Chroma implementations
│   ├── retriever.py         # Retrieval logic with chunk size variants
│   └── evaluation.py        # Quality metrics logging
├── data/                    # Sample documents for testing
├── notebooks/
│   └── rag_pipeline.ipynb   # Interactive RAG tutorial
└── README.md                # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install langchain langchain-openai langchain-community
pip install faiss-cpu chromadb pypdf tiktoken
pip install pandas numpy scikit-learn
```

### Run the RAG Pipeline

```bash
# Start Jupyter and open the notebook
jupyter notebook notebooks/rag_pipeline.ipynb
```

## 📚 Core Components

### 1. Document Loader

```python
from rag.document_loader import DocumentProcessor

processor = DocumentProcessor()
docs = processor.load_pdf("data/sample.pdf")
chunks = processor.create_chunks(docs, chunk_size=1000, overlap=200)
```

### 2. Vector Store

```python
from rag.vector_store import VectorStoreManager

# FAISS
vs_faiss = VectorStoreManager.create_faiss(chunks, embeddings)

# Chroma
vs_chroma = VectorStoreManager.create_chroma(chunks, embeddings, persist_dir="./db")
```

### 3. Retriever

```python
from rag.retriever import ChunkSizeComparisonRetriever

retriever = ChunkSizeComparisonRetriever(embeddings)
results = retriever.compare_chunk_sizes(docs, query, [500, 1000, 1500])
```

### 4. Evaluation

```python
from rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()
metrics = evaluator.evaluate_retrieval(query, retrieved_docs, ground_truth)
```

## 🔧 Chunk Size Comparison

The module includes a comparison of different chunk sizes:

```python
for chunk_size in [500, 1000, 1500]:
    overlap = int(chunk_size * 0.2)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    texts = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(texts, embeddings)
```

| Chunk Size | Pros | Cons |
|------------|------|------|
| 500 | Better precision, focused context | More chunks, may lose broader context |
| 1000 | Balanced approach | May include some irrelevant info |
| 1500 | More context per chunk | Lower precision, higher token cost |

## 📊 Evaluation Metrics

### Retrieval Metrics

- **Precision@K**: Relevant docs in top K results
- **Recall@K**: Proportion of all relevant docs retrieved
- **MRR**: Mean Reciprocal Rank of first relevant doc
- **NDCG**: Normalized Discounted Cumulative Gain

### Generation Metrics

- **Answer Relevance**: Semantic similarity to query
- **Faithfulness**: Alignment with retrieved context
- **Latency**: Response time
- **Token Usage**: Cost tracking

## 🧪 Testing with Sample Documents

Create sample documents in `data/`:

```bash
mkdir -p data
echo "Your test content here" > data/sample1.txt
```

Or use provided sample data generator:

```python
from rag.document_loader import create_sample_documents

create_sample_documents("data/")
```

## 📖 RAG Pipeline Steps

1. **Ingest**: Load documents from various sources
2. **Chunk**: Split into optimal-sized pieces
3. **Embed**: Convert to vector representations
4. **Index**: Store in vector database
5. **Retrieve**: Find relevant context for queries
6. **Generate**: Produce answers using retrieved context
7. **Evaluate**: Measure quality and performance

## 🎯 Best Practices

### Chunking Strategy

```python
# Consider these factors:
- Document structure (paragraphs, sections)
- Model context window
- Query complexity
- Desired granularity

# Default starting point:
chunk_size = 1000
chunk_overlap = 200  # 20%
```

### Vector Store Selection

| Store | Best For | Persistence |
|-------|----------|-------------|
| FAISS | Large-scale, fast search | Manual save/load |
| Chroma | Development, metadata filtering | Built-in persistence |
| Pinecone | Production, cloud scale | Managed |

### Retrieval Optimization

```python
# Hybrid search
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Fetch more, then diversify
        "lambda_mult": 0.5  # Balance relevance/diversity
    }
)
```

## 📝 Exercises

1. **Chunk Size Experiment**
   - Test 3 different chunk sizes on the same documents
   - Measure retrieval precision for each
   - Plot performance vs. chunk size

2. **Vector Store Comparison**
   - Compare FAISS vs Chroma for:
     - Indexing speed
     - Query latency
     - Memory usage

3. **Evaluation Suite**
   - Create 10 test questions with ground truth
   - Measure Precision@3, Recall@5, MRR
   - Document findings

4. **Advanced Retrieval**
   - Implement query expansion
   - Add re-ranking with cross-encoder
   - Compare results

## 🔗 Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chroma Documentation](https://docs.trychroma.com/)
- [RAG Evaluation Paper](https://arxiv.org/abs/2305.14283)

## ✅ Week 11 Completion Checklist

- [ ] Understand RAG architecture
- [ ] Implement document loading and chunking
- [ ] Create FAISS and Chroma vector stores
- [ ] Build retriever with chunk size comparison
- [ ] Implement evaluation metrics
- [ ] Test with sample documents
- [ ] Measure and log latency and token usage
- [ ] Compare chunk size performance
- [ ] Document optimization findings

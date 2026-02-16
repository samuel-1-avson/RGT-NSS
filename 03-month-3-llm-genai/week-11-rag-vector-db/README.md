# Week 11: RAG & Vector Databases

> **Branch**: `week-11-rag-vector-db` | **Review Required**: Yes  
> **Dataset**: Custom corpus (public domain articles)

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-11-rag-vector-db
git push origin week-11-rag-vector-db
```

---

## Learning Objectives
- Understand embeddings and vector search
- Implement chunking strategies
- Build RAG pipelines
- Compare vector database options

---

## Dataset

**Source**: Custom corpus (Wikipedia articles, public domain books)  
**Format**: Text files  
**Topics**: Machine Learning, Data Science, AI

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Read Pinecone RAG overview
- [ ] Review Weaviate Academy quickstart

### Guided Lab (≤120 min)

#### Lab 11.1: RAG Pipeline
```python
# rag/pipeline.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_rag_pipeline(documents_path, chunk_size=1000):
    """Build RAG pipeline with FAISS."""
    
    # Load documents
    loader = TextLoader(documents_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size * 0.2
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save index
    vectorstore.save_local('data/faiss_index')
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
        chain_type='stuff',
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )
    
    return qa_chain

# Test different chunk sizes
for chunk_size in [500, 1000, 1500]:
    qa = build_rag_pipeline('data/corpus.txt', chunk_size)
    result = qa.run("What is machine learning?")
    print(f"Chunk size {chunk_size}: {result[:100]}...")
```

#### Lab 11.2: Retrieval Quality Metrics
```python
# rag/evaluation.py
def evaluate_retrieval(vectorstore, queries, expected_docs):
    """Evaluate retrieval quality."""
    metrics = []
    
    for query, expected in zip(queries, expected_docs):
        retrieved = vectorstore.similarity_search(query, k=5)
        retrieved_texts = [doc.page_content for doc in retrieved]
        
        # Check if expected doc is in retrieved
        hit = any(expected in text for text in retrieved_texts)
        metrics.append({
            'query': query,
            'hit': hit,
            'retrieved_count': len(retrieved)
        })
    
    hit_rate = sum(m['hit'] for m in metrics) / len(metrics)
    print(f"Hit Rate: {hit_rate:.2%}")
    
    return metrics
```

### Independent Work (≤120 min)
- [ ] Build complete RAG pipeline
- [ ] Log retrieval quality metrics
- [ ] Create evaluation scripts

---

## Deliverable

**RAG Pipeline** (`rag/`) with:
- Document ingestion
- Vector index (FAISS)
- Retrieval and generation
- Quality metrics

---

## Folder Structure
```
week-11-rag-vector-db/
├── rag/
│   ├── pipeline.py
│   └── evaluation.py
├── data/
│   ├── corpus.txt
│   └── faiss_index/
└── README.md
```

---

## Commit Message
```
week-11: Add RAG pipeline with FAISS and chunking experiments

- Implement document loading and chunking
- Create FAISS vector index
- Test different chunk sizes (500, 1000, 1500)
- Add retrieval quality evaluation
```

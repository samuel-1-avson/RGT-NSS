# Week 10: Building LLM Apps with LangChain

## 📖 Overview

This week focuses on building production-ready LLM applications using LangChain. You'll learn to create document Q&A systems, manage chains and agents, and integrate vector databases.

## 🎯 Learning Objectives

- Understand LangChain architecture (chains, agents, memory)
- Build document Q&A applications
- Implement text splitting and chunking strategies
- Integrate vector stores for retrieval
- Write tests for LLM applications

## 📁 Directory Structure

```
week-10-langchain-apps/
├── app/
│   ├── qa_app.py           # Main Q&A application
│   └── utils.py            # Helper functions
├── notebooks/
│   └── langchain_intro.ipynb  # Interactive tutorial
├── tests/
│   └── test_app.py         # Unit tests
├── data/                   # Sample documents (user-provided)
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install langchain langchain-openai langchain-community
pip install chromadb pypdf python-dotenv
pip install pytest pytest-asyncio
```

### Environment Setup

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key" > .env
```

### Run the Q&A App

```bash
# Place a PDF in the data/ directory, then:
python app/qa_app.py --pdf data/sample.pdf
```

## 📚 Core Components

### 1. Document Loading

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('document.pdf')
docs = loader.load()
```

### 2. Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = splitter.split_documents(docs)
```

### 3. Vector Store

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=OpenAIEmbeddings()
)
```

### 4. Retrieval QA Chain

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app
```

## 📖 Key Concepts

### Chains

Chains combine multiple components into a single pipeline:

```python
from langchain.chains import LLMChain, SimpleSequentialChain

# Create individual chains
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine into sequential chain
overall_chain = SimpleSequentialChain(chains=[chain1, chain2])
```

### Document Loaders

LangChain supports multiple document formats:

| Loader | Use Case |
|--------|----------|
| `PyPDFLoader` | PDF documents |
| `TextLoader` | Plain text files |
| `CSVLoader` | CSV data |
| `JSONLoader` | JSON documents |
| `UnstructuredHTMLLoader` | HTML pages |

### Text Splitters

| Splitter | Best For |
|----------|----------|
| `RecursiveCharacterTextSplitter` | General text (recommended) |
| `CharacterTextSplitter` | Simple character-based splitting |
| `TokenTextSplitter` | Token-based splitting |
| `MarkdownHeaderTextSplitter` | Markdown documents |

### Vector Stores

| Store | Type | Best For |
|-------|------|----------|
| `Chroma` | Local | Development, small datasets |
| `FAISS` | Local | Large datasets, fast search |
| `Pinecone` | Cloud | Production, scale |
| `Weaviate` | Cloud | Multi-modal, GraphQL |

## 🔧 Best Practices

1. **Chunk Size Selection**
   - Smaller chunks (500-1000): Better precision, more context needed
   - Larger chunks (1000-2000): More context, may include irrelevant info
   - Overlap 10-20% to maintain context across chunks

2. **Error Handling**
   - Always wrap API calls in try-except blocks
   - Implement retry logic with exponential backoff
   - Validate inputs before processing

3. **Cost Management**
   - Track token usage for each operation
   - Use cheaper models (gpt-3.5-turbo) for simple tasks
   - Cache embeddings to avoid redundant API calls

4. **Testing**
   - Mock LLM calls in unit tests
   - Test document loading separately from chain logic
   - Use fixtures for test data

## 📝 Exercises

1. **Enhance the Q&A App**
   - Add support for multiple PDFs
   - Implement conversation memory
   - Add source citations to answers

2. **Custom Chain**
   - Build a chain that summarizes then translates
   - Add custom prompts for specific domains

3. **Evaluation**
   - Create a test set of questions and expected answers
   - Measure retrieval accuracy
   - Compare different chunk sizes

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook.ipynb)

## ✅ Week 10 Completion Checklist

- [ ] Understand LangChain core abstractions
- [ ] Build and run the Q&A application
- [ ] Test with different PDF documents
- [ ] Implement text splitting variations
- [ ] Write unit tests for components
- [ ] Compare Chroma vs FAISS performance
- [ ] Measure token usage and costs
- [ ] Submit completed application

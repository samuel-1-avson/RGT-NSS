# Week 10: Building LLM Apps with LangChain

> **Branch**: `week-10-langchain-apps` | **Review Required**: Yes  
> **Dataset**: PDF Documents (public domain)

---

## Git Workflow
```bash
git checkout main
git pull origin main
git checkout -b week-10-langchain-apps
git push origin week-10-langchain-apps
```

---

## Learning Objectives
- Build chains with LangChain
- Implement memory and conversation
- Use tools and agents appropriately
- Add observability and logging

---

## Dataset

**Source**: Project Gutenberg (public domain books)  
**Format**: PDF or text files  
**Example**: [Alice in Wonderland](https://www.gutenberg.org/files/11/11-0.txt)

---

## Weekly Structure

### Prep (≤60 min)
- [ ] Read LangChain introduction
- [ ] Complete LangChain tutorials

### Guided Lab (≤120 min)

#### Lab 10.1: Q&A over Documents
```python
# app/qa_app.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Load document
loader = TextLoader('data/alice_in_wonderland.txt')
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    texts, 
    embeddings, 
    persist_directory='data/chroma'
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
    chain_type='stuff',
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query
query = "Who is the Queen of Hearts?"
result = qa_chain({"query": query})

print(f"Answer: {result['result']}")
print(f"\nSources:")
for doc in result['source_documents']:
    print(f"- {doc.page_content[:100]}...")
```

#### Lab 10.2: Add Logging
```python
# app/logging_utils.py
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_query(query, response, source_docs, start_time):
    """Log query details."""
    elapsed = time.time() - start_time
    logger.info(f"Query: {query}")
    logger.info(f"Response time: {elapsed:.2f}s")
    logger.info(f"Sources used: {len(source_docs)}")
```

### Independent Work (≤120 min)
- [ ] Instrument app with logging
- [ ] Add unit tests
- [ ] Document setup

---

## Deliverable

**LangChain App** (`app/`) with:
- Q&A functionality
- Logging and observability
- Unit tests
- Setup instructions

---

## Folder Structure
```
week-10-langchain-apps/
├── app/
│   ├── qa_app.py
│   └── logging_utils.py
├── data/
│   └── alice_in_wonderland.txt
├── tests/
│   └── test_qa.py
└── README.md
```

---

## Commit Message
```
week-10: Add LangChain Q&A app with PDF support

- Implement document loading and chunking
- Create vector store with Chroma
- Add retrieval QA chain
- Implement logging for observability
```

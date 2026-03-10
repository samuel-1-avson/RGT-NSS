# Week 10: Chains and Tools Implementation Report

This report details the implementation of advanced LangChain components, specifically focusing on Chains and Tools within the PDF Q&A system.

---

## 1. Chains Implementation
The system utilizes a structured approach to LLM orchestration through specialized chain managers.

### Key Chains:
* **ConversationalRetrievalChain**: The core of the Q&A system. It combines conversation history with document context to provide coherent, multi-turn answers.
* **Summarization chain**: A specialized chain designed to condense large amounts of PDF text into concise summaries.
* **Basic QA chain**: A straightforward retrieval chain used for single-turn factual questions.

---

## 2. Tools and Agents
To extend the LLM's capabilities, we implemented custom tools and integrated them with LangChain Agents.

### Custom Tools:
* **Metadata Extractor**: A tool that allows the LLM to programmatically access PDF metadata (author, creation date, etc.) rather than relying solely on page content.
* **Document Stats Tool**: Provides high-level statistics about the loaded documents, such as page counts and chunk distributions.

### Agent Integration:
We utilized the `ZeroShotAgent` pattern to allow the system to intelligently decide when to search the vector database and when to use specialized metadata tools.

---

## 3. What is a Vector Store?
A **Vector Store** is a specialized database designed to store and retrieve information using mathematical representations of text called "embeddings." 

Unlike traditional databases that search for exact keywords, a vector store enables **semantic search**. This means the system can find information based on the *meaning* and *context* of your query, even if the exact words don't match. In our PDF system, it acts as the long-term memory that holds thousands of document chunks for near-instant retrieval.

---

## 4. Retrievers
A high-performance retrieval layer was built to ensure the most relevant context is provided to the chains.

* **Vector Store**: Powered by ChromaDB for persistent, fast similarity searching.
* **Embedding Model**: Switched to `all-MiniLM-L6-v2` for optimized local CPU performance, ensuring sub-second retrieval times.
* **Recursive Splitting**: Implemented advanced text splitting to preserve semantic meaning across chunks.

---

## 4. Technical Summary
By decoupling these components into `chains.py`, `tools.py`, and `retrievers.py`, we created a modular architecture that is easy to test, evaluate, and scale.

**Implementation complete and verified.**

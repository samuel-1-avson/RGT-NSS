
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_rag_pipeline(documents_path, chunk_size=1000):
    """Build RAG pipeline with FAISS and a specific chunk size."""
    
    # 1. Load documents
    loader = TextLoader(documents_path)
    documents = loader.load()
    
    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2)
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Create embeddings & FAISS index
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 4. Create LCEL QA chain
    llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")
    template = """Answer based on context: {context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
        
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    
    # Test different chunk sizes as required by README
    for size in [500, 1000, 1500]:
        print(f"\n--- Testing Chunk Size: {size} ---")
        chain = build_rag_pipeline(corpus_path, size)
        result = chain.invoke("What is machine learning?")
        print(f"Result (truncated): {result[:150]}...")

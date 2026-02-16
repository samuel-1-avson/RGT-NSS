
import os
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.logging_utils import log_query, logger

def run_qa_app(query_text):
    """Main function to run Q&A over Alice in Wonderland using LCEL."""
    start_time = time.time()
    
    # 1. Load document
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "alice_in_wonderland.txt")
    loader = TextLoader(data_path)
    documents = loader.load()
    
    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings & Vector Store
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 4. Initialize LLM
    llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")
    
    # 5. Build LCEL Chain
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Execute Query
    logger.info(f"Processing query: {query_text}")
    # For LCEL, we get source docs separately to log them
    source_docs = retriever.invoke(query_text)
    answer = rag_chain.invoke(query_text)
    
    # 7. Log and Return
    log_query(query_text, answer, source_docs, start_time)
    
    return {
        "result": answer,
        "source_documents": source_docs
    }

if __name__ == "__main__":
    query = "Who stole the tarts and what happened?"
    res = run_qa_app(query)
    print(f"\nQUERY: {query}")
    print(f"ANSWER: {res['result']}")

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retrieval import retrieve_context

def create_rag_chain():
    """Create a RAG chain for answering questions."""
    
    # Template emphasizing citations
    template = """You are a helpful Telecom Policy Assistant. Answer the question based ONLY on the following context.
    
    If the answer is not in the context, say "I don't have enough information in the policy documents to answer that."
    
    Always cite the source document name for your answer.
    
    Context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize LLM (Ollama)
    llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")
    
    def format_docs(docs):
        return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs)
    
    rag_chain = (
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

async def generate_answer(question: str):
    """Retrieves documents and generates an answer."""
    docs = retrieve_context(question)
    chain = create_rag_chain()
    
    response = await chain.ainvoke({"context": docs, "question": question})
    
    return {
        "answer": response,
        "sources": [d.metadata.get("source") for d in docs],
        "context": [d.page_content for d in docs]
    }

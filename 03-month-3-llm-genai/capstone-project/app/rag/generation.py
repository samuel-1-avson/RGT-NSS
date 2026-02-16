from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .retrieval import retrieve_context
from typing import List, Dict, Any

# Initialize Global LLM (Ollama)
llm = ChatOllama(model="llama3", temperature=0, base_url="http://127.0.0.1:11434")

def format_docs(docs):
    """Format documents with source metadata."""
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs)

def create_rag_chain():
    """Create a standard RAG chain."""
    template = """You are a helpful Telecom Policy Assistant. Answer the question based ONLY on the following context.
    
    If the answer is not in the context, say "I don't have enough information in the policy documents to answer that."
    
    Always cite the source document name for your answer.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": lambda x: x["context"], "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

async def generate_multi_queries(question: str) -> List[str]:
    """Generate 3 variations of the question to improve retrieval."""
    template = """You are an AI assistant tasked with generating three different versions of the given user 
    question to retrieve relevant documents from a vector database. By generating multiple perspectives 
    on the user query, your goal is to help the user overcome some of the limitations of the distance-based 
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = await chain.ainvoke({"question": question})
    queries = [q.strip() for q in response.split("\n") if q.strip()][:3]
    return [question] + queries

async def generate_hypothetical_answer(question: str) -> str:
    """HyDE: Generate a hypothetical answer to improve retrieval."""
    template = """Please write a short hypothetical paragraph from a telecom policy document that answers this question. 
    Focus on the technical and legal language typically found in such documents.
    Question: {question}
    Hypothetical Policy snippet:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return await chain.ainvoke({"question": question})

async def verify_answer(question: str, context: str, answer: str) -> str:
    """Verified: Self-correction step to ensure faithfulness to context."""
    template = """You are a quality controller. Compare the provided Answer against the Context.
    
    1. Identify any claims in the Answer not supported by the Context.
    2. If there are hallucinations, rewrite the Answer to be 100% faithful to the Context.
    3. Ensure citations are preserved.
    4. If the Answer is already perfect, just return the original Answer.
    
    Question: {question}
    Context: {context}
    Original Answer: {answer}
    
    Verified Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return await chain.ainvoke({"question": question, "context": context, "answer": answer})

async def generate_answer(question: str, strategy: str = "simple"):
    """Retrieves documents and generates an answer using specified strategy."""
    
    # 1. RETRIEVAL STEP
    if strategy == "multi_query":
        queries = await generate_multi_queries(question)
        all_docs = []
        for q in queries:
            all_docs.extend(retrieve_context(q, k=2))
        # Unique docs
        docs = {d.metadata.get('source', '') + d.page_content: d for d in all_docs}.values()
        docs = list(docs)[:5]
    elif strategy == "hyde":
        hypothetical_doc = await generate_hypothetical_answer(question)
        docs = retrieve_context(hypothetical_doc, k=4)
    else:
        docs = retrieve_context(question, k=4)

    # 2. GENERATION STEP
    context_str = format_docs(docs)
    chain = create_rag_chain()
    answer = await chain.ainvoke({"context": context_str, "question": question})

    # 3. VERIFICATION STEP (Optional)
    if strategy == "verified":
        answer = await verify_answer(question, context_str, answer)

    return {
        "answer": answer,
        "strategy_used": strategy,
        "sources": list(set([d.metadata.get("source") for d in docs])),
        "context": [d.page_content for d in docs]
    }

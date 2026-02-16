
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def evaluate_retrieval(vectorstore, queries, expected_keywords):
    """Evaluate retrieval quality using Hit Rate."""
    metrics = []
    
    for query, expected in zip(queries, expected_keywords):
        # Retrieve top 3
        retrieved = vectorstore.similarity_search(query, k=3)
        retrieved_texts = [doc.page_content.lower() for doc in retrieved]
        
        # Check if expected keyword is in retrieved texts
        hit = any(expected.lower() in text for text in retrieved_texts)
        metrics.append({
            'query': query,
            'hit': hit,
            'retrieved_count': len(retrieved)
        })
    
    hit_rate = sum(m['hit'] for m in metrics) / len(metrics)
    print(f"Hit Rate: {hit_rate:.2%}")
    
    return metrics

if __name__ == "__main__":
    # Test logic
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    
    corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    loader = TextLoader(corpus_path)
    texts = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(loader.load())
    
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://127.0.0.1:11434")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    queries = ["Tell me about FAISS", "What is vector search?"]
    expected = ["Facebook AI Similarity Search", "Similarity Search"]
    
    evaluate_retrieval(vectorstore, queries, expected)

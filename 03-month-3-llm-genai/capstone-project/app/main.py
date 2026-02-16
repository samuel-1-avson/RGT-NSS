from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Capstone Policy Assistant API",
    version="1.0.0",
    description="RAG-based API for Telecom Policy Queries"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3005",
        "http://localhost:3006",
        "http://localhost:8000", 
        "http://localhost:8001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "capstone-rag-api"}

@app.post("/chat")
async def chat_endpoint(query: str, strategy: str = "simple"):
    """Chat endpoint for Q&A with policy documents using selected strategy."""
    from app.rag.generation import generate_answer
    try:
        response = await generate_answer(query, strategy=strategy)
        return response
    except Exception as e:
        return {"error": str(e)}

@app.post("/ingest")
async def trigger_ingestion():
    """Manually trigger document ingestion (REBUILD)."""
    from app.rag.ingestion import load_documents, split_documents
    from app.rag.retrieval import add_documents_to_db, FAISS_DB_DIR
    import shutil
    
    try:
        # Clear existing DB to prevent duplicates
        if os.path.exists(FAISS_DB_DIR):
            shutil.rmtree(FAISS_DB_DIR)
            
        docs = load_documents()
        chunks = split_documents(docs)
        add_documents_to_db(chunks)
        return {"message": f"Successfully rebuilt knowledge base with {len(chunks)} chunks."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new policy document."""
    import shutil
    from app.rag.ingestion import DOCS_DIR
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        
    file_path = os.path.join(DOCS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"message": f"Successfully uploaded {file.filename}", "path": file_path}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/documents")
async def list_documents():
    """List all available policy documents."""
    from app.rag.ingestion import DOCS_DIR
    
    if not os.path.exists(DOCS_DIR):
        return {"documents": []}
        
    try:
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md") or f.endswith(".pdf")]
        return {"documents": files}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/admin/documents/{filename}")
async def delete_document(filename: str):
    """Delete a policy document."""
    from app.rag.ingestion import DOCS_DIR
    
    file_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
        
    try:
        os.remove(file_path)
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Trigger initial ingestion if needed (simplified for dev)
    # trigger_ingestion()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)

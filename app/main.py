"""
FastAPI Application — Document Q&A API with RAG + PostgreSQL
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR
from app.rag import process_pdf, ask_question, search_similar, get_stats
from app.database import (
    init_db, get_db,
    save_document, get_all_documents, delete_all_documents_db,
    delete_document_by_filename,
    save_qa, get_qa_history, clear_qa_history
)


# ============================================================
# Pydantic Schemas
# ============================================================

class QuestionRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Document Q&A API",
    description="Upload PDFs and ask questions using RAG (Retrieval Augmented Generation)",
    version="2.0.0",
)

@app.on_event("startup")
def startup():
    init_db()


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the web interface."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r") as f:
        return f.read()


@app.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}


@app.get("/stats")
def system_stats():
    """Get system info: loaded documents, provider, model."""
    return get_stats()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a PDF document, process it, and save to database."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        stats = process_pdf(file_path)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    save_document(db, filename=file.filename, pages=stats["pages"], chunks=stats["chunks"])

    return {
        "message": f"'{file.filename}' uploaded and processed successfully.",
        "filename": file.filename,
        "pages": stats["pages"],
        "chunks": stats["chunks"],
    }


@app.post("/ask")
def ask(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question about the uploaded documents. Saves Q&A to database."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask_question(request.question)

    save_qa(db, question=request.question, answer=result["answer"], sources=result["sources"])

    return result


@app.post("/search")
def search(request: SearchRequest):
    """Search for similar document chunks (without LLM)."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = search_similar(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results}


@app.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent question-answer history from the database."""
    return get_qa_history(db, limit=limit)


@app.delete("/history")
def delete_history(db: Session = Depends(get_db)):
    """Clear all Q&A history."""
    clear_qa_history(db)
    return {"message": "Q&A history cleared."}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents from the database."""
    return get_all_documents(db)


@app.delete("/documents/{filename}")
def delete_single_document(filename: str, db: Session = Depends(get_db)):
    """Delete a single document by filename."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    # Remove file
    os.remove(file_path)

    # Remove from database
    delete_document_by_filename(db, filename)

   # Rebuild vectorstore from remaining PDFs
    import app.rag as rag_module
    rag_module._vector_store = None
    rag_module._qa_chain = None

    # Clear vectorstore contents (don't delete the folder — Docker volume mount)
    vectorstore_dir = "vectorstore"
    if os.path.exists(vectorstore_dir):
        for f in os.listdir(vectorstore_dir):
            fp = os.path.join(vectorstore_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    # Re-process remaining PDFs
    remaining_pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    for pdf in remaining_pdfs:
        rag_module.process_pdf(os.path.join(UPLOAD_DIR, pdf))

    return {"message": f"'{filename}' deleted successfully.", "remaining": len(remaining_pdfs)}


@app.delete("/documents")
def delete_all_documents(db: Session = Depends(get_db)):
    """Delete all uploaded documents, clear vector store, and clear database."""
    for f in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    vectorstore_dir = "vectorstore"
    if os.path.exists(vectorstore_dir):
        for f in os.listdir(vectorstore_dir):
            fp = os.path.join(vectorstore_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    import app.rag as rag_module
    rag_module._vector_store = None
    rag_module._qa_chain = None

    delete_all_documents_db(db)

    return {"message": "All documents, vector store, and database records cleared."}
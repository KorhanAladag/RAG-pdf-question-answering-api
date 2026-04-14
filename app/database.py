"""
Database — PostgreSQL connection, table definitions, and CRUD functions.
Uses SQLAlchemy for ORM and connection management.
"""

from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

from app.config import DATABASE_URL

# ============================================================
# Connection Setup
# ============================================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ============================================================
# Table Definitions (SQLAlchemy models)
# ============================================================

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    pages = Column(Integer, default=0)
    chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    qa_history = relationship("QAHistory", back_populates="document")


class QAHistory(Base):
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text, default="[]")
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    asked_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="qa_history")


# ============================================================
# Create Tables
# ============================================================

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


# ============================================================
# Database Session (dependency for FastAPI)
# ============================================================

def get_db():
    """Yield a database session, close it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# CRUD Functions — Documents
# ============================================================

def save_document(db, filename: str, pages: int, chunks: int):
    """Save a new document record to the database."""
    doc = Document(filename=filename, pages=pages, chunks=chunks)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_all_documents(db):
    """Get all uploaded documents, newest first."""
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "pages": doc.pages,
            "chunks": doc.chunks,
            "uploaded_at": doc.uploaded_at.isoformat(),
        }
        for doc in docs
    ]


def get_document_by_filename(db, filename: str):
    """Find a document by its filename."""
    return db.query(Document).filter(Document.filename == filename).first()


def delete_document_by_filename(db, filename: str):
    """Delete a single document and its QA history by filename."""
    doc = db.query(Document).filter(Document.filename == filename).first()
    if doc:
        db.query(QAHistory).filter(QAHistory.document_id == doc.id).delete()
        db.delete(doc)
        db.commit()
        return True
    return False


def delete_all_documents_db(db):
    """Delete all documents and their QA history."""
    db.query(QAHistory).delete()
    db.query(Document).delete()
    db.commit()


# ============================================================
# CRUD Functions — QA History
# ============================================================

def save_qa(db, question: str, answer: str, sources: list, document_id: int = None):
    """Save a question-answer pair to the database."""
    qa = QAHistory(
        question=question,
        answer=answer,
        sources=json.dumps(sources),
        document_id=document_id,
    )
    db.add(qa)
    db.commit()
    db.refresh(qa)
    return qa


def get_qa_history(db, limit: int = 50):
    """Get recent QA history, newest first."""
    records = (
        db.query(QAHistory)
        .order_by(QAHistory.asked_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "sources": json.loads(r.sources),
            "asked_at": r.asked_at.isoformat(),
        }
        for r in records
    ]


def clear_qa_history(db):
    """Delete all QA history records."""
    db.query(QAHistory).delete()
    db.commit()

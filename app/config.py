"""
Configuration — All settings and environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpass@localhost:5432/ragdb"
)

# --- Embedding Model ---
# Free, runs locally, no API key needed
# For Turkish PDFs use: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM Provider ---
# Options: "anthropic", "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Ollama (free, local)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- RAG Settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# --- Create directories ---
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

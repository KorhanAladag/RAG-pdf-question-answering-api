# PDF Question Answering API

A RAG (Retrieval Augmented Generation) powered API that lets you upload PDF documents and ask questions about their content. Features a web interface with document management, Q&A history, and individual file deletion. Built with **FastAPI**, **LangChain**, **FAISS**, **PostgreSQL**, and **Docker**.

## How It Works

```
PDF Upload → Extract Text → Split into Chunks → Generate Embeddings → Store in FAISS
                                                                          ↓
User Question → Generate Embedding → Find Similar Chunks → Send to LLM → Answer
                                                                          ↓
                                                              Save Q&A to PostgreSQL
```

1. Upload a PDF through the web UI or API
2. The PDF is split into overlapping text chunks (1000 chars, 200 overlap)
3. Each chunk is converted to a 384-dimensional vector using Sentence Transformers
4. Vectors are indexed in FAISS for fast similarity search
5. When you ask a question, the 4 most relevant chunks are retrieved and sent to the LLM as context
6. The question, answer, and sources are saved to PostgreSQL for history tracking

## Features

- **PDF Upload** — drag & drop or browse, with processing stats (pages, chunks)
- **Question Answering** — RAG-based answers with source citations (file + page number)
- **Individual Document Deletion** — delete specific PDFs with the ✕ button, vectorstore auto-rebuilds
- **Bulk Document Deletion** — clear all documents, vectorstore, and database at once
- **Q&A History** — recent questions and answers stored in PostgreSQL, viewable in the UI
- **Clear History** — one-click button to wipe Q&A history
- **Similarity Search** — debug endpoint to see raw retrieval results without LLM
- **Multi-language Support** — answers in the same language as the question
- **LLM Flexibility** — switch between Anthropic Claude (API) and Ollama (free, local) via config
- **Swagger Docs** — auto-generated interactive API documentation at `/docs`

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| RAG Pipeline | LangChain |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Anthropic Claude / Ollama (local) |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy |
| Containerization | Docker Compose (multi-container) |
| PDF Parsing | PyPDF |

## Quick Start

### Option 1: Docker Compose (Recommended)

Runs both the API and PostgreSQL in containers with one command.

```bash
git clone https://github.com/yourusername/pdf-question-answering-api.git
cd pdf-question-answering-api

cp .env.example .env
# Edit .env — add your Anthropic API key

docker compose up --build

```
### Running without an API key (free, local)

1. Install Ollama: https://ollama.com/download
2. Pull a model in terminal: `ollama pull llama3.1`
3. In your `.env` file, set:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
4. Run: `docker compose up --build`

Open http://localhost:8000 for the web UI or http://localhost:8000/docs for Swagger API docs.

### Option 2: Local Development

Uses conda for Python and Docker only for PostgreSQL.

```bash
git clone https://github.com/yourusername/pdf-question-answering-api.git
cd pdf-question-answering-api

# Start PostgreSQL only
docker compose up db -d

# Create environment
conda create -n rag python=3.12 -y
conda activate rag
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env — add your Anthropic API key
# Change DATABASE_URL to: postgresql://raguser:ragpass@localhost:5432/ragdb

# Run the API
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System info (documents loaded, LLM provider, model) |
| `POST` | `/upload` | Upload a PDF (multipart form) |
| `POST` | `/ask` | Ask a question (JSON body) — saves to database |
| `POST` | `/search` | Similarity search without LLM (debug) |
| `GET` | `/documents` | List all uploaded documents from database |
| `DELETE` | `/documents/{filename}` | Delete a specific document and rebuild vectorstore |
| `DELETE` | `/documents` | Delete all documents, vectorstore, and database records |
| `GET` | `/history` | Get Q&A history from database |
| `DELETE` | `/history` | Clear all Q&A history |
### Example Usage (cURL)

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'

# List uploaded documents
curl http://localhost:8000/documents

# Delete a specific document
curl -X DELETE http://localhost:8000/documents/document.pdf

# View Q&A history
curl http://localhost:8000/history

# Clear Q&A history
curl -X DELETE http://localhost:8000/history

# Similarity search (no LLM, for debugging)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 3}'

# Delete everything
curl -X DELETE http://localhost:8000/documents
```

### Example Usage (Python)

```python
import requests

BASE = "http://localhost:8000"

# Upload
files = {"file": open("paper.pdf", "rb")}
requests.post(f"{BASE}/upload", files=files)

# Ask
response = requests.post(f"{BASE}/ask", json={"question": "What are the main findings?"})
print(response.json()["answer"])

# History
history = requests.get(f"{BASE}/history").json()
for h in history:
    print(f"Q: {h['question']}\nA: {h['answer'][:100]}...\n")

# Delete specific document
requests.delete(f"{BASE}/documents/paper.pdf")
```

## Configuration

All settings are managed through environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `ollama` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DATABASE_URL` | `postgresql://raguser:ragpass@db:5432/ragdb` | PostgreSQL connection string |

### Using Ollama (Free, Local — no data leaves your machine)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1

# Set in .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Database Schema

```
documents
├── id (PK, auto-increment)
├── filename (string, not null)
├── pages (integer)
├── chunks (integer)
└── uploaded_at (datetime, auto)

qa_history
├── id (PK, auto-increment)
├── question (text, not null)
├── answer (text, not null)
├── sources (JSON string)
├── document_id (FK → documents.id, nullable)
└── asked_at (datetime, auto)
```

## Project Structure

```
pdf-question-answering-api/
├── app/
│   ├── __init__.py          # Python module marker
│   ├── config.py            # Environment variables & settings
│   ├── database.py          # PostgreSQL tables & CRUD functions
│   ├── main.py              # FastAPI endpoints (11 routes)
│   ├── rag.py               # RAG pipeline (load → chunk → embed → query)
│   └── templates/
│       └── index.html       # Web UI (upload, Q&A, history, delete buttons)
├── uploads/                  # Uploaded PDFs (gitignored)
├── vectorstore/              # FAISS index (gitignored)
├── requirements.txt          # Python dependencies
├── Dockerfile                # API container image
├── docker-compose.yml        # Multi-container setup (API + PostgreSQL)
├── .env.example              # Environment variable template
├── .gitignore                # Files excluded from git
└── README.md
```

## License

MIT

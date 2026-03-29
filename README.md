# PDF Question Answering API

A RAG (Retrieval Augmented Generation) powered API that lets you upload PDF documents and ask questions about their content. Built with **FastAPI**, **LangChain**, **FAISS**, **PostgreSQL**, and **Docker**.

## How It Works

```
PDF Upload → Extract Text → Split into Chunks → Generate Embeddings → Store in FAISS
                                                                          ↓
User Question → Generate Embedding → Find Similar Chunks → Send to LLM → Answer
                                                                          ↓
                                                              Save Q&A to PostgreSQL
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| RAG Pipeline | LangChain |
| Vector Store | FAISS |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | Anthropic Claude / Ollama (local) |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Containerization | Docker Compose |
| PDF Parsing | PyPDF |

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
git clone https://github.com/yourusername/pdf-question-answering-api.git
cd pdf-question-answering-api

cp .env.example .env
# Edit .env with your API key

docker compose up --build
```

Open http://localhost:8000 for the web UI or http://localhost:8000/docs for Swagger API docs.

### Option 2: Local Development

```bash
git clone https://github.com/yourusername/pdf-question-answering-api.git
cd pdf-question-answering-api

# Create virtual environment
conda create -n rag python=3.12 -y
conda activate rag

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API key

# Start PostgreSQL (requires Docker)
docker compose up db -d

# Run the API
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System info |
| `POST` | `/upload` | Upload a PDF |
| `POST` | `/ask` | Ask a question |
| `POST` | `/search` | Similarity search (no LLM) |
| `GET` | `/history` | Q&A history from database |
| `GET` | `/documents` | List uploaded documents from database |
| `DELETE` | `/documents` | Delete all data |

### Example Usage (cURL)

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'

# View Q&A history
curl http://localhost:8000/history

# List documents
curl http://localhost:8000/documents
```

### Example Usage (Python)

```python
import requests

# Upload
files = {"file": open("paper.pdf", "rb")}
requests.post("http://localhost:8000/upload", files=files)

# Ask
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are the main findings?"}
)
print(response.json()["answer"])

# History
history = requests.get("http://localhost:8000/history").json()
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `ollama` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `DATABASE_URL` | `postgresql://raguser:ragpass@db:5432/ragdb` | PostgreSQL connection |

### Using Ollama (Free, Local)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# Set in .env
LLM_PROVIDER=ollama
```

## Database Schema

```
documents
├── id (PK)
├── filename
├── pages
├── chunks
└── uploaded_at

qa_history
├── id (PK)
├── question
├── answer
├── sources (JSON)
├── document_id (FK → documents.id)
└── asked_at
```

## Project Structure

```
pdf-question-answering-api/
├── app/
│   ├── __init__.py
│   ├── config.py            # Environment variables & settings
│   ├── database.py          # PostgreSQL tables & CRUD functions
│   ├── main.py              # FastAPI endpoints
│   ├── rag.py               # RAG pipeline
│   └── templates/
│       └── index.html       # Web UI
├── uploads/                  # Uploaded PDFs (gitignored)
├── vectorstore/              # FAISS index (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT

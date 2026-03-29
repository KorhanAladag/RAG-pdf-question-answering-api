"""
RAG Pipeline — PDF loading, chunking, embedding, vector storage, and querying.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.config import (
    EMBEDDING_MODEL, LLM_PROVIDER, ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL, VECTORSTORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)

# ============================================================
# Module-level cache — initialized once, reused across requests
# ============================================================
_embeddings = None
_vector_store = None
_qa_chain = None


def get_embeddings():
    """Load embedding model (runs locally, no API key needed)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def load_pdf(file_path: str) -> list:
    """Read a PDF file and return list of page documents."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def split_into_chunks(documents: list, chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_or_update_vectorstore(chunks: list) -> FAISS:
    """Create a new FAISS vector store from chunks, or merge into existing one."""
    global _vector_store
    embeddings = get_embeddings()

    new_store = FAISS.from_documents(chunks, embeddings)

    if _vector_store is None:
        index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
        if os.path.exists(index_path):
            _vector_store = FAISS.load_local(
                VECTORSTORE_DIR, embeddings,
                allow_dangerous_deserialization=True
            )
            _vector_store.merge_from(new_store)
        else:
            _vector_store = new_store
    else:
        _vector_store.merge_from(new_store)

    _vector_store.save_local(VECTORSTORE_DIR)
    return _vector_store


def get_vectorstore() -> FAISS:
    """Load existing vectorstore from disk."""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
        if os.path.exists(index_path):
            _vector_store = FAISS.load_local(
                VECTORSTORE_DIR, embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            return None
    return _vector_store


def create_llm():
    """Create LLM based on configured provider."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=1024,
        )
    elif LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def build_qa_chain():
    """Build the full RAG chain: retriever + prompt + LLM."""
    global _qa_chain

    vector_store = get_vectorstore()
    if vector_store is None:
        return None

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = PromptTemplate(
        template="""Use the following context to answer the question.
If the answer is not in the context, say "I couldn't find this information in the uploaded documents."
Do not make up information.
Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    llm = create_llm()

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return _qa_chain


def process_pdf(file_path: str) -> dict:
    """Full pipeline: PDF → pages → chunks → vectorstore."""
    pages = load_pdf(file_path)
    chunks = split_into_chunks(pages)
    create_or_update_vectorstore(chunks)

    global _qa_chain
    _qa_chain = None

    return {
        "pages": len(pages),
        "chunks": len(chunks),
    }


def ask_question(question: str) -> dict:
    """Query the RAG chain with a question."""
    global _qa_chain

    if _qa_chain is None:
        _qa_chain = build_qa_chain()

    if _qa_chain is None:
        return {
            "answer": "No documents uploaded yet. Please upload a PDF first.",
            "sources": []
        }

    result = _qa_chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", -1),
            "preview": doc.page_content[:200]
        })

    return {
        "answer": result["result"],
        "sources": sources
    }


def search_similar(query: str, top_k: int = TOP_K) -> list:
    """Search for similar chunks without LLM — useful for debugging."""
    vector_store = get_vectorstore()
    if vector_store is None:
        return []

    results = vector_store.similarity_search_with_score(query, k=top_k)

    return [
        {
            "content": doc.page_content[:300],
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", -1),
            "score": float(score)
        }
        for doc, score in results
    ]


def get_stats() -> dict:
    """Get current system stats."""
    vector_store = get_vectorstore()

    uploaded_files = []
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        uploaded_files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]

    return {
        "documents_loaded": len(uploaded_files),
        "vectorstore_ready": vector_store is not None,
        "llm_provider": LLM_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "files": uploaded_files
    }

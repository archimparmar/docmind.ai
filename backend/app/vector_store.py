"""
FAISS vector store management.

FAISS (Facebook AI Similarity Search) runs entirely in-process and
persists to plain files on disk — no server, no account, no recurring
cost. The trade-off vs. a managed vector DB like Pinecone: no built-in
multi-tenant filtering, metadata querying, or horizontal scaling across
machines. For a single-user / small-document-set project, that trade-off
is the right one.

We persist the index to disk after every ingest so it survives server
restarts, and reload it lazily on first use.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.embedding_model,
    google_api_key=settings.google_api_key,
)

_vector_store: FAISS | None = None


def _index_files_exist() -> bool:
    return (settings.index_dir / "index.faiss").exists()


def get_vector_store() -> FAISS | None:
    """Return the loaded vector store, or None if nothing has been ingested yet."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if _index_files_exist():
        _vector_store = FAISS.load_local(
            str(settings.index_dir),
            _embeddings,
            allow_dangerous_deserialization=True,  # safe: it's our own local file, not user-supplied
        )
    return _vector_store


def add_documents(chunks: list[Document]) -> None:
    """Embed and add chunks to the store, creating it on first use."""
    global _vector_store
    store = get_vector_store()

    if store is None:
        _vector_store = FAISS.from_documents(chunks, _embeddings)
    else:
        store.add_documents(chunks)
        _vector_store = store

    _vector_store.save_local(str(settings.index_dir))


def has_documents() -> bool:
    return get_vector_store() is not None

"""
The RAG (Retrieval-Augmented Generation) chain.

Deliberately written explicitly rather than via a one-line LangChain
helper, so every step is visible:
  1. embed the user's question
  2. retrieve the k most similar chunks from FAISS
  3. stuff those chunks into a prompt as "context"
  4. ask Gemini to answer using ONLY that context

This "stuff" strategy is the simplest RAG pattern: it works well as long
as k * chunk_size fits comfortably in the model's context window. For
much larger document sets you'd reach for map-reduce or re-ranking
instead — worth knowing the limitation, not just the happy path.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.vector_store import get_vector_store

SYSTEM_PROMPT = """You are DocMind AI, an assistant that answers questions \
strictly using the provided document context.

Rules:
- Only use information present in the context below.
- If the answer isn't in the context, say so plainly — do not guess or use outside knowledge.
- Be concise and cite the page number(s) you used, like (p. 3).

Context:
{context}
"""

_llm = ChatGoogleGenerativeAI(
    model=settings.chat_model,
    google_api_key=settings.google_api_key,
    temperature=0.2,  # low temperature: favor grounded, repeatable answers over creativity
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")
        parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class NoDocumentsError(Exception):
    """Raised when a question is asked before any document has been ingested."""


def answer_question(question: str) -> dict:
    """
    Run the full retrieve -> prompt -> generate pipeline for one question.

    Returns a dict with the answer text and the source chunks used, so the
    API (and frontend) can show users which pages backed the answer.
    """
    store = get_vector_store()
    if store is None:
        raise NoDocumentsError("No documents have been uploaded yet.")

    retriever = store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    retrieved_docs = retriever.invoke(question)

    context = _format_context(retrieved_docs)
    formatted_prompt = _prompt.invoke({"context": context, "question": question})

    response = _llm.invoke(formatted_prompt)

    sources = [
        {
            "source": doc.metadata.get("source", "document"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200],
        }
        for doc in retrieved_docs
    ]

    return {"answer": response.content, "sources": sources}

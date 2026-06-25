"""
Document ingestion: load a PDF from disk and split it into overlapping
chunks suitable for embedding.

Why chunk at all? Embedding models and the LLM's context window both have
limits, and retrieval works best over small, semantically coherent pieces
of text rather than whole documents. RecursiveCharacterTextSplitter tries
paragraph breaks first, then sentences, then words — so chunks stay as
semantically clean as possible instead of cutting mid-sentence.
"""

from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def _load_pages(file_path: Path) -> list[Document]:
    """
    Extract text page-by-page using pdfplumber.

    Note: we deliberately use pdfplumber instead of LangChain's
    PyPDFLoader (which wraps pypdf). pypdf mis-parses some PDFs that use
    CID/Identity-H encoded fonts (common in PDFs exported from Canva,
    Figma, and similar design tools) — it inserts a space between every
    letter, e.g. "P R O F E S S I O N A L" instead of "PROFESSIONAL".
    pdfplumber's extraction handles these fonts correctly. Always sanity
    check extracted text on a sample PDF before trusting a new loader.
    """
    pages = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": i},  # 0-indexed, matches PyPDFLoader convention
                )
            )
    return pages


def load_and_split_pdf(file_path: Path, source_name: str) -> list[Document]:
    """
    Load a PDF and split it into chunks.

    Each resulting chunk keeps metadata pointing back to its source file
    and original page number, so answers can later be traced to a page.
    """
    pages = _load_pages(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Tag every chunk with the source filename for citation in answers
    for chunk in chunks:
        chunk.metadata["source"] = source_name

    return chunks

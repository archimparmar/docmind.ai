"""
FastAPI app exposing two endpoints:
  POST /upload  -> ingest a PDF into the vector store
  POST /ask     -> ask a question, answered via RAG over ingested docs

Run with: uvicorn app.main:app --reload --port 8000
"""

import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingestion import load_and_split_pdf
from app.rag_chain import NoDocumentsError, answer_question
from app.vector_store import add_documents, has_documents

app = FastAPI(title="DocMind AI", description="RAG-based PDF question answering")

# Streamlit runs on a different port, so CORS must be opened for local dev.
# For a real deployment, lock allow_origins down to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents_loaded": has_documents()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save with a unique name to avoid collisions between users/uploads
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = settings.upload_dir / safe_name

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = load_and_split_pdf(dest_path, source_name=file.filename)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in this PDF (it may be scanned/image-only).",
            )
        add_documents(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "message": "Document ingested successfully.",
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(request.question)
    except NoDocumentsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc

    return result

# DocMind AI — Intelligent Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
uploaded PDF documents, grounded entirely in the document's content.

**Stack:** Python · FastAPI · LangChain · Gemini API · FAISS · Streamlit

Built to run at **zero recurring cost**: Gemini's free tier handles
embeddings and chat, and FAISS stores vectors locally on disk — no paid
vector database, no OpenAI billing.

## How it works

```
PDF upload
   │
   ▼
pdfplumber extracts text per page
   │
   ▼
RecursiveCharacterTextSplitter splits into ~1000-char overlapping chunks
   │
   ▼
Gemini embedding model (text-embedding-004) embeds each chunk
   │
   ▼
FAISS index stores the vectors locally (data/faiss_index/)
   │
   ▼
User asks a question
   │
   ▼
Question is embedded → FAISS returns the k most similar chunks
   │
   ▼
Chunks are "stuffed" into a prompt as context
   │
   ▼
Gemini chat model (gemini-2.0-flash) answers using ONLY that context
   │
   ▼
Answer + source chunks (with page numbers) returned to the user
```

This is the simplest RAG pattern ("stuff" retrieval). It works well as
long as `k × chunk_size` comfortably fits in the model's context window.
For much larger document collections, re-ranking or map-reduce
summarization would be the next step.

## Project structure

```
docmind-ai/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes: /upload, /ask, /health
│   │   ├── config.py        # env var loading & validation (pydantic-settings)
│   │   ├── ingestion.py     # PDF text extraction + chunking
│   │   ├── vector_store.py  # FAISS index management (create/load/persist)
│   │   └── rag_chain.py     # retrieval + prompt + Gemini generation
│   ├── data/
│   │   ├── uploads/         # uploaded PDFs (gitignored)
│   │   └── faiss_index/     # persisted vector index (gitignored)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── app.py                # Streamlit chat UI
    └── requirements.txt
```

## Setup

### 1. Get a free Gemini API key
Visit [Google AI Studio](https://aistudio.google.com/app/apikey) → sign in
→ "Create API key".

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GOOGLE_API_KEY

uvicorn app.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

### 3. Frontend

In a second terminal:

```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Opens at `http://localhost:8501`. Upload a PDF in the sidebar, then ask
questions in the chat box.

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status + whether any document is loaded |
| `/upload` | POST | Upload a PDF (multipart form, field name `file`) |
| `/ask` | POST | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |

## Design notes / things worth knowing

- **Why FAISS over Pinecone/a managed vector DB:** zero cost, no account,
  no network dependency for search. Trade-off: no built-in multi-tenant
  filtering or horizontal scaling — fine for a single-user project, not
  for a multi-user SaaS product without extra work.
- **Why pdfplumber instead of LangChain's PyPDFLoader (pypdf):** some
  PDFs (notably ones exported from Canva, Figma, and similar tools) use
  CID/Identity-H encoded fonts. `pypdf`'s text extraction misreads word
  boundaries on these and inserts a space between every letter.
  `pdfplumber` handles them correctly — confirmed by testing against a
  real-world PDF with this exact font encoding.
- **Why "stuff" retrieval and not something fancier:** simplest correct
  RAG pattern, easy to reason about and explain. The known limitation is
  context-window size with very large document sets — worth naming
  proactively rather than pretending it scales infinitely.
- **Grounding:** the system prompt explicitly instructs the model to
  answer only from retrieved context and say so if the answer isn't
  there, rather than fall back on general knowledge — this is what makes
  it a RAG chatbot rather than just "an LLM with extra text pasted in."

## Possible extensions

- Multi-document support with per-document filtering at query time
- Conversation memory (currently each question is answered independently)
- Dockerize both services for one-command startup
- Swap FAISS for a hosted vector DB if multi-user support is needed

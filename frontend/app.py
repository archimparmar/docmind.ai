"""
DocMind AI — Streamlit frontend.

Talks to the FastAPI backend over HTTP. Kept deliberately simple: a file
uploader, an upload status area, and a chat-style Q&A loop. Streamlit
re-runs this whole script on every interaction, so chat history is held
in st.session_state to survive reruns.
"""

import requests
import streamlit as st

BACKEND_URL = st.secrets.get(
    "BACKEND_URL",
    "https://docmind-ai-5b7h.onrender.com"
)

st.set_page_config(page_title="DocMind AI", page_icon="📄", layout="centered")
st.title("📄 DocMind AI")
st.caption("Upload a PDF, then ask questions about it — answered with Gemini + RAG over FAISS.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

# --- Upload section ---
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest document", type="primary"):
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Added {data['chunks_added']} chunks from {data['filename']}")
                        st.session_state.doc_uploaded = True
                    else:
                        st.error(response.json().get("detail", "Upload failed."))
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the backend. Is `uvicorn app.main:app` running on port 8000?")

    st.divider()
    st.caption("Stack: FastAPI · LangChain · Gemini · FAISS")

# --- Chat section ---
for question, answer, sources in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources used"):
                for src in sources:
                    page = src.get("page")
                    page_display = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**{src['source']}**, page {page_display}")
                    st.caption(src["snippet"] + "...")

question = st.chat_input("Ask a question about your document")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask", json={"question": question}, timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    st.write(answer)
                    if sources:
                        with st.expander("Sources used"):
                            for src in sources:
                                page = src.get("page")
                                page_display = page + 1 if isinstance(page, int) else page
                                st.markdown(f"**{src['source']}**, page {page_display}")
                                st.caption(src["snippet"] + "...")
                    st.session_state.chat_history.append((question, answer, sources))
                else:
                    error_msg = response.json().get("detail", "Something went wrong.")
                    st.error(error_msg)
                    st.session_state.chat_history.append((question, f"Error: {error_msg}", []))
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Is `uvicorn app.main:app` running on port 8000?")

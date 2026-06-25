"""
Centralised app configuration.

Using pydantic-settings means env vars are validated once, at startup,
instead of scattered os.getenv() calls everywhere. If GOOGLE_API_KEY is
missing, the app fails fast with a clear error instead of failing deep
inside a LangChain call later.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str

    # Gemini models — flash is fast + free-tier friendly for both chat and embeddings
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    retrieval_k: int = 4

    # Storage paths
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "faiss_index"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

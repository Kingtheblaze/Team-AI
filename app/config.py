"""
ChronoStream RAG - Configuration Settings
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "ChronoStream RAG"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEMO_PORT: int = 8501

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    DEVICE: str = "cpu"  # 'cpu', 'cuda', or 'mps'

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "in_memory"  # 'in_memory', 'chroma', 'qdrant'
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: int = 6333

    # Hierarchical Adaptive Prototype Settings (Bounded Memory)
    MAX_ACTIVE_PROTOTYPES: int = 50
    PROTOTYPE_SIMILARITY_THRESHOLD: float = 0.82
    TEMPORAL_DECAY_HALF_LIFE_SEC: float = 300.0  # 5 minutes half-life
    MAX_MEMORY_CHUNKS: int = 500  # Strict upper bound on raw chunk retention
    CHUNK_RETENTION_MINUTES: int = 60  # Raw chunks older than this are pruned or compacted

    # Hybrid Retrieval Weights
    WEIGHT_DENSE: float = 0.50
    WEIGHT_SPARSE: float = 0.25
    WEIGHT_RECENCY: float = 0.25
    RECENCY_DECAY_RATE: float = 0.005  # Per-second decay for retrieval re-ranking

    # LLM Settings
    LLM_PROVIDER: str = "fast_extractive"  # 'fast_extractive', 'ollama', 'openai', 'vllm'
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "llama3.2:3b"

    # Stream Simulator Settings
    STREAM_INTERVAL_SECONDS: float = 2.0
    DEFAULT_STREAM_SOURCE: str = "stream://live-news-intel"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

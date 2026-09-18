"""
Vector Store & Embedding Manager with Temporal Indexing and Bounded Memory.
Supports sentence-transformers with instant zero-dependency fallback.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np

from app.config import settings
from app.ingestion.chunker import StreamChunk
from app.utils.logger import get_logger
from app.utils.time_utils import now_utc

logger = get_logger("VectorStore")


class EmbeddingEngine:
    """
    Manages vector embeddings. Uses sentence-transformers when available,
    falling back to a fast hash-based pseudo-semantic embedder so the system
    is guaranteed runnable on any machine immediately.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, dimension: int = settings.EMBEDDING_DIMENSION):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._is_transformer = False
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info(f"Loading SentenceTransformer: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=settings.DEVICE)
            self._is_transformer = True
            logger.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.warning(
                f"SentenceTransformer not initialized ({e}). Using deterministic token-hash embedding engine."
            )
            self._is_transformer = False

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized vector embedding for a single text."""
        if self._is_transformer and self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()

        # Fallback deterministic pseudo-semantic embedding (bag-of-words / hash projection)
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            h = hash(w) % self.dimension
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class TemporalVectorStore:
    """
    Time-aware in-memory vector database with bounded sliding retention.
    Never requires full index rebuilds; handles continuous O(1) incremental inserts.
    """

    def __init__(
        self,
        max_chunks: int = settings.MAX_MEMORY_CHUNKS,
        retention_minutes: int = settings.CHUNK_RETENTION_MINUTES,
    ):
        self.max_chunks = max_chunks
        self.retention_minutes = retention_minutes
        self.chunks: List[StreamChunk] = []
        self.embeddings: List[np.ndarray] = []

    def add_chunk(self, chunk: StreamChunk, embedding: List[float]):
        """Incremental upsert of a single chunk and its embedding."""
        emb_arr = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm

        chunk.embedding = emb_arr.tolist()
        self.chunks.append(chunk)
        self.embeddings.append(emb_arr)

        # Enforce bounded retention
        if len(self.chunks) > self.max_chunks:
            self._evict_oldest()

    def _evict_oldest(self):
        """Bounded memory: evict oldest items past capacity."""
        overflow = len(self.chunks) - self.max_chunks
        self.chunks = self.chunks[overflow:]
        self.embeddings = self.embeddings[overflow:]

    def prune_by_retention(self, reference_time: Optional[datetime] = None):
        """Prune chunks older than the retention horizon."""
        ref = reference_time or now_utc()
        cutoff = ref - timedelta(minutes=self.retention_minutes)

        filtered_chunks = []
        filtered_embeddings = []
        for ch, emb in zip(self.chunks, self.embeddings):
            if ch.timestamp >= cutoff:
                filtered_chunks.append(ch)
                filtered_embeddings.append(emb)

        self.chunks = filtered_chunks
        self.embeddings = filtered_embeddings

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Tuple[StreamChunk, float]]:
        """
        Dense vector similarity search with hard temporal filtering.
        """
        if not self.chunks:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        candidates: List[Tuple[StreamChunk, float]] = []

        for ch, emb in zip(self.chunks, self.embeddings):
            # Hard temporal bounds filter
            if start_time and ch.timestamp < start_time:
                continue
            if end_time and ch.timestamp > end_time:
                continue

            sim = float(np.dot(q_vec, emb))
            candidates.append((ch, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_stored_chunks": len(self.chunks),
            "max_capacity": self.max_chunks,
            "retention_window_minutes": self.retention_minutes,
        }

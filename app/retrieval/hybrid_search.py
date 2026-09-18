"""
Hybrid Time-Aware Search Engine.
Combines Dense Semantic Embeddings + Sparse BM25 Term Matching + Temporal Recency Bias.
"""
from datetime import datetime
from typing import List, Tuple, Optional
import math
from collections import Counter
import numpy as np

from app.config import settings
from app.ingestion.chunker import StreamChunk
from app.utils.time_utils import now_utc


def compute_bm25_sparse_score(query_tokens: List[str], doc_tokens: List[str], avg_doc_len: float = 20.0) -> float:
    """Lightweight BM25 term saturation scoring function."""
    k1 = 1.5
    b = 0.75
    doc_len = len(doc_tokens)
    if doc_len == 0:
        return 0.0

    counts = Counter(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = counts.get(qt, 0)
        if tf > 0:
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += numerator / denominator
    return min(1.0, score / (len(query_tokens) + 1e-5))


class HybridTimeAwareSearcher:
    """
    Ranks streaming chunks using a unified formula:
    Score = w_dense * DenseSim + w_sparse * BM25Score + w_recency * RecencyFactor
    """

    def __init__(
        self,
        weight_dense: float = settings.WEIGHT_DENSE,
        weight_sparse: float = settings.WEIGHT_SPARSE,
        weight_recency: float = settings.WEIGHT_RECENCY,
        recency_decay_rate: float = settings.RECENCY_DECAY_RATE,
    ):
        self.w_dense = weight_dense
        self.w_sparse = weight_sparse
        self.w_recency = weight_recency
        self.recency_decay_rate = recency_decay_rate

    def rank(
        self,
        query: str,
        query_embedding: List[float],
        candidates: List[Tuple[StreamChunk, float]],
        reference_time: Optional[datetime] = None,
    ) -> List[Tuple[StreamChunk, float, dict]]:
        """
        Takes candidate chunks and dense scores, computes hybrid time-aware re-ranking.
        Returns sorted list of (chunk, final_score, score_breakdown).
        """
        ref = reference_time or now_utc()
        query_tokens = [w.lower() for w in query.split() if len(w) > 2]

        ranked: List[Tuple[StreamChunk, float, dict]] = []

        for chunk, dense_sim in candidates:
            doc_tokens = [w.lower() for w in chunk.text.split()]
            sparse_score = compute_bm25_sparse_score(query_tokens, doc_tokens)

            # Calculate temporal recency score
            age_seconds = max(0.0, (ref - chunk.timestamp).total_seconds())
            recency_score = math.exp(-self.recency_decay_rate * age_seconds)

            # Normalize dense_sim from [-1, 1] to [0, 1]
            dense_norm = max(0.0, (dense_sim + 1.0) / 2.0)

            final_score = (
                self.w_dense * dense_norm
                + self.w_sparse * sparse_score
                + self.w_recency * recency_score
            )

            breakdown = {
                "dense_score": round(dense_norm, 4),
                "sparse_score": round(sparse_score, 4),
                "recency_score": round(recency_score, 4),
                "age_seconds": round(age_seconds, 1),
                "final_score": round(final_score, 4),
            }

            ranked.append((chunk, final_score, breakdown))

        # Sort descending by final hybrid score
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

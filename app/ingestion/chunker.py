"""
Chunking and Data Models for Streaming Ingestion.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
from app.utils.time_utils import now_utc, to_iso_utc


@dataclass
class StreamChunk:
    id: str
    text: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def timestamp_iso(self) -> str:
        return to_iso_utc(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp_iso,
            "source": self.source,
            "metadata": self.metadata,
        }


class StreamChunker:
    """
    Lightweight streaming chunker that segments text into temporally grounded units.
    Ensures every emitted chunk retains a strictly tracked UTC timestamp.
    """

    def __init__(self, max_chunk_chars: int = 400, overlap_chars: int = 50):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_text(
        self,
        text: str,
        source: str = "stream://live-feed",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[StreamChunk]:
        """Split a piece of incoming text into one or more StreamChunk instances."""
        if not text or not text.strip():
            return []

        ts = timestamp or now_utc()
        meta = metadata or {}
        text = text.strip()

        # If short enough, emit single chunk directly
        if len(text) <= self.max_chunk_chars:
            return [
                StreamChunk(
                    id=str(uuid.uuid4())[:8],
                    text=text,
                    timestamp=ts,
                    source=source,
                    metadata=meta,
                )
            ]

        # Break text into chunks respecting word boundaries
        words = text.split()
        chunks: List[StreamChunk] = []
        curr_words: List[str] = []
        curr_len = 0

        for word in words:
            if curr_len + len(word) + 1 > self.max_chunk_chars and curr_words:
                chunk_str = " ".join(curr_words)
                chunks.append(
                    StreamChunk(
                        id=str(uuid.uuid4())[:8],
                        text=chunk_str,
                        timestamp=ts,
                        source=source,
                        metadata=meta,
                    )
                )
                # Apply word overlap
                overlap_words = curr_words[-3:] if len(curr_words) >= 3 else curr_words
                curr_words = list(overlap_words)
                curr_len = sum(len(w) + 1 for w in curr_words)

            curr_words.append(word)
            curr_len += len(word) + 1

        if curr_words:
            chunk_str = " ".join(curr_words)
            chunks.append(
                StreamChunk(
                    id=str(uuid.uuid4())[:8],
                    text=chunk_str,
                    timestamp=ts,
                    source=source,
                    metadata=meta,
                )
            )

        return chunks

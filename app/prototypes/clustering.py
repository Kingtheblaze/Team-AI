"""
Online Micro-Clustering for Hierarchical Adaptive Prototypes.
Maintains bounded memory representation of streaming text embeddings.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import math
import numpy as np

from app.utils.time_utils import now_utc, to_iso_utc


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two unit-normalized vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class MicroCluster:
    """
    Adaptive prototype micro-cluster representing a semantic topic thread over time.
    """
    id: str
    label: str
    centroid: List[float]
    count: int = 1
    first_seen: datetime = field(default_factory=now_utc)
    last_updated: datetime = field(default_factory=now_utc)
    exemplars: List[str] = field(default_factory=list)  # Key representative sentences
    sources: List[str] = field(default_factory=list)
    salience_score: float = 1.0

    def add_point(self, embedding: List[float], text: str, source: str, timestamp: datetime):
        """Incrementally update centroid with online running average."""
        c = np.asarray(self.centroid, dtype=np.float32)
        e = np.asarray(embedding, dtype=np.float32)

        # Online running average for centroid
        new_c = (c * self.count + e) / (self.count + 1)
        norm = np.linalg.norm(new_c)
        if norm > 0:
            new_c = new_c / norm
        self.centroid = new_c.tolist()

        self.count += 1
        self.last_updated = timestamp
        if source not in self.sources:
            self.sources.append(source)

        # Keep top representative exemplars (bounded at 5)
        if len(self.exemplars) < 5:
            self.exemplars.append(text)
        else:
            # Replace oldest exemplar
            self.exemplars.pop(0)
            self.exemplars.append(text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "count": self.count,
            "salience_score": round(self.salience_score, 4),
            "first_seen": to_iso_utc(self.first_seen),
            "last_updated": to_iso_utc(self.last_updated),
            "exemplars": self.exemplars,
            "sources": self.sources,
        }

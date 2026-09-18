"""
In-Memory Temporal Knowledge Graph.
Maintains timestamped entity-relation networks with temporal slicing.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from app.ingestion.chunker import StreamChunk
from app.temporal_kg.entity_extractor import TemporalEntityExtractor
from app.utils.logger import get_logger
from app.utils.time_utils import to_iso_utc, now_utc

logger = get_logger("TemporalKG")


@dataclass
class TemporalEdge:
    subject: str
    predicate: str
    object: str
    timestamp: datetime
    source_id: str
    source_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "timestamp": to_iso_utc(self.timestamp),
            "source_id": self.source_id,
            "source_url": self.source_url,
        }


class TemporalKnowledgeGraph:
    """
    Lightweight, event-driven in-memory Temporal Knowledge Graph.
    Supports incremental edge upserting, bounded temporal window filtering,
    and entity path retrieval.
    """

    def __init__(self, max_edges: int = 1000):
        self.max_edges = max_edges
        self.extractor = TemporalEntityExtractor()
        self.edges: List[TemporalEdge] = []
        self.entity_index: Dict[str, List[int]] = defaultdict(list)

    def add_chunk(self, chunk: StreamChunk):
        """Extract triplets from incoming stream chunk and incrementally upsert edges."""
        triplets = self.extractor.extract_triplets(chunk.text)
        for sub, pred, obj in triplets:
            edge = TemporalEdge(
                subject=sub,
                predicate=pred,
                object=obj,
                timestamp=chunk.timestamp,
                source_id=chunk.id,
                source_url=chunk.source,
            )
            self.edges.append(edge)
            edge_idx = len(self.edges) - 1
            self.entity_index[sub.lower()].append(edge_idx)
            self.entity_index[obj.lower()].append(edge_idx)

        # Enforce bounded capacity on edges
        if len(self.edges) > self.max_edges:
            self._prune_old_edges()

    def _prune_old_edges(self):
        """Keep only the most recent max_edges."""
        trim_count = len(self.edges) - self.max_edges
        self.edges = self.edges[trim_count:]
        # Rebuild index
        self.entity_index.clear()
        for idx, edge in enumerate(self.edges):
            self.entity_index[edge.subject.lower()].append(idx)
            self.entity_index[edge.object.lower()].append(idx)

    def query_window(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[TemporalEdge]:
        """Return all relationships observed within a given UTC time window."""
        results = []
        for edge in self.edges:
            if start_time and edge.timestamp < start_time:
                continue
            if end_time and edge.timestamp > end_time:
                continue
            results.append(edge)
        return results

    def query_entity(self, entity_name: str, start_time: Optional[datetime] = None) -> List[TemporalEdge]:
        """Return all chronological facts related to a specific entity."""
        name_lower = entity_name.lower().strip()
        matching_indices = self.entity_index.get(name_lower, [])
        results = []
        for idx in matching_indices:
            if idx < len(self.edges):
                edge = self.edges[idx]
                if start_time and edge.timestamp < start_time:
                    continue
                results.append(edge)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Return nodes and edges summary formatted for graph visualizers."""
        nodes: Set[str] = set()
        edge_dicts = []

        for e in self.edges[-50:]:  # Last 50 recent edges
            nodes.add(e.subject)
            nodes.add(e.object)
            edge_dicts.append(e.to_dict())

        return {
            "total_edges": len(self.edges),
            "unique_entities": len(self.entity_index),
            "recent_nodes": list(nodes),
            "recent_edges": edge_dicts,
        }

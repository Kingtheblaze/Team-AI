"""
Hierarchical Adaptive Prototype Manager.
Coordinates micro-clusters, heavy-hitters tracking, temporal decay, and bounded memory pruning.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np

from app.config import settings
from app.prototypes.clustering import MicroCluster, cosine_similarity
from app.prototypes.decay import TemporalDecay
from app.utils.logger import get_logger
from app.utils.time_utils import now_utc

logger = get_logger("PrototypeManager")


class HierarchicalPrototypeManager:
    """
    Manages online hierarchical prototypes for streaming text chunks.
    Ensures memory footprint remains bounded while maintaining high semantic resolution.
    """

    def __init__(
        self,
        max_prototypes: int = settings.MAX_ACTIVE_PROTOTYPES,
        similarity_threshold: float = settings.PROTOTYPE_SIMILARITY_THRESHOLD,
        decay_half_life_sec: float = settings.TEMPORAL_DECAY_HALF_LIFE_SEC,
    ):
        self.max_prototypes = max_prototypes
        self.similarity_threshold = similarity_threshold
        self.decay = TemporalDecay(half_life_seconds=decay_half_life_sec)
        self.prototypes: Dict[str, MicroCluster] = {}
        self._total_ingested = 0

    def assimilate_chunk(
        self,
        embedding: List[float],
        text: str,
        source: str,
        timestamp: Optional[datetime] = None,
    ) -> MicroCluster:
        """
        Assimilate an incoming vector into the adaptive prototype hierarchy.
        Performs either cluster update or new prototype emergence.
        """
        ts = timestamp or now_utc()
        self._total_ingested += 1

        best_proto: Optional[MicroCluster] = None
        best_sim = -1.0

        # Find closest existing prototype
        for proto in self.prototypes.values():
            sim = cosine_similarity(embedding, proto.centroid)
            if sim > best_sim:
                best_sim = sim
                best_proto = proto

        # If sufficiently similar to an existing prototype, merge
        if best_proto is not None and best_sim >= self.similarity_threshold:
            best_proto.add_point(embedding, text, source, ts)
            best_proto.salience_score = self.decay.compute_salience_score(
                count=best_proto.count,
                last_updated=best_proto.last_updated,
                reference_time=ts,
            )
            return best_proto

        # Otherwise, spawn a new micro-cluster prototype
        proto_id = f"proto-{str(uuid.uuid4())[:6]}"
        # Generate short readable label from text prefix
        label = " ".join(text.split()[:5]) + ("..." if len(text.split()) > 5 else "")
        new_proto = MicroCluster(
            id=proto_id,
            label=label,
            centroid=list(embedding),
            count=1,
            first_seen=ts,
            last_updated=ts,
            exemplars=[text],
            sources=[source],
            salience_score=self.decay.compute_salience_score(count=1, last_updated=ts, reference_time=ts),
        )
        self.prototypes[proto_id] = new_proto

        # Check memory bounds and enforce pruning/merging if needed
        self._enforce_memory_bounds(reference_time=ts)
        return new_proto

    def _enforce_memory_bounds(self, reference_time: Optional[datetime] = None):
        """
        Maintains bounded memory. When active prototypes exceed max limit,
        prunes lowest salience (decayed) clusters or merges closest pairs.
        """
        if len(self.prototypes) <= self.max_prototypes:
            return

        ref = reference_time or now_utc()
        # Recalculate all salience scores
        for proto in self.prototypes.values():
            proto.salience_score = self.decay.compute_salience_score(
                count=proto.count, last_updated=proto.last_updated, reference_time=ref
            )

        # Sort by salience ascending (lowest salience first)
        sorted_protos = sorted(self.prototypes.values(), key=lambda p: p.salience_score)

        # Evict lowest salience prototype
        victim = sorted_protos[0]
        logger.info(
            f"Bounded memory limit reached ({len(self.prototypes)}/{self.max_prototypes}). "
            f"Pruning stale prototype {victim.id} ('{victim.label}')"
        )
        del self.prototypes[victim.id]

    def query_prototypes(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        reference_time: Optional[datetime] = None,
    ) -> List[Tuple[MicroCluster, float]]:
        """
        Retrieve the most relevant active prototypes for a given query vector.
        Weights similarity by current temporal salience.
        """
        if not self.prototypes:
            return []

        ref = reference_time or now_utc()
        results: List[Tuple[MicroCluster, float]] = []

        for proto in self.prototypes.values():
            sim = cosine_similarity(query_embedding, proto.centroid)
            # Re-evaluate dynamic salience with decay
            salience = self.decay.compute_salience_score(
                count=proto.count, last_updated=proto.last_updated, reference_time=ref
            )
            # Hybrid prototype rank score
            score = 0.75 * sim + 0.25 * min(1.0, salience)
            results.append((proto, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics on memory footprint and prototype states."""
        return {
            "total_active_prototypes": len(self.prototypes),
            "max_allowed_prototypes": self.max_prototypes,
            "total_chunks_ingested": self._total_ingested,
            "prototypes": [p.to_dict() for p in sorted(self.prototypes.values(), key=lambda x: x.salience_score, reverse=True)],
        }

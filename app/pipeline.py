"""
ChronoStream RAG - Core Pipeline Engine.
Orchestrates ingestion → embedding → prototype assimilation → KG update → retrieval → generation.
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.config import settings
from app.ingestion import StreamSimulator, StreamChunk
from app.prototypes import HierarchicalPrototypeManager
from app.temporal_kg import TemporalKnowledgeGraph
from app.retrieval import EmbeddingEngine, TemporalVectorStore, HybridTimeAwareSearcher, RetrievalTrigger
from app.generation import AnswerGenerator
from app.utils.logger import get_logger
from app.utils.time_utils import now_utc, to_iso_utc

logger = get_logger("Pipeline")


class ChronoStreamPipeline:
    """
    Central orchestrator. Wires all subsystems together into a single
    continuous streaming RAG pipeline.
    """

    def __init__(self):
        logger.info("Initializing ChronoStream RAG Pipeline...")
        self.embedder = EmbeddingEngine()
        self.vector_store = TemporalVectorStore()
        self.prototype_manager = HierarchicalPrototypeManager()
        self.temporal_kg = TemporalKnowledgeGraph()
        self.hybrid_searcher = HybridTimeAwareSearcher()
        self.retrieval_trigger = RetrievalTrigger()
        self.generator = AnswerGenerator()
        self.stream_simulator = StreamSimulator(interval_seconds=settings.STREAM_INTERVAL_SECONDS)
        self._ingestion_task: Optional[asyncio.Task] = None
        self._chunks_ingested = 0
        logger.info("Pipeline initialized. All subsystems ready.")

    async def start_ingestion(self):
        """Launch the continuous background ingestion loop."""
        if self._ingestion_task and not self._ingestion_task.done():
            logger.warning("Ingestion already running.")
            return
        self._ingestion_task = asyncio.create_task(self._ingest_loop())
        logger.info("Background ingestion loop started.")

    async def _ingest_loop(self):
        """Core ingestion event loop: stream → chunk → embed → assimilate → KG."""
        try:
            async for chunk in self.stream_simulator.stream_generator():
                await self.ingest_chunk(chunk)
        except asyncio.CancelledError:
            logger.info("Ingestion loop cancelled.")
        except Exception as e:
            logger.error(f"Ingestion loop error: {e}", exc_info=True)

    async def ingest_chunk(self, chunk: StreamChunk):
        """Process a single incoming chunk through the full ingestion pipeline."""
        # 1. Generate embedding
        embedding = self.embedder.embed_text(chunk.text)

        # 2. Incremental vector store upsert (never full rebuild)
        self.vector_store.add_chunk(chunk, embedding)

        # 3. Assimilate into hierarchical adaptive prototypes
        proto = self.prototype_manager.assimilate_chunk(
            embedding=embedding,
            text=chunk.text,
            source=chunk.source,
            timestamp=chunk.timestamp,
        )

        # 4. Incremental temporal knowledge graph update
        self.temporal_kg.add_chunk(chunk)

        self._chunks_ingested += 1
        if self._chunks_ingested % 5 == 0:
            logger.info(
                f"Ingested {self._chunks_ingested} chunks | "
                f"VectorStore: {len(self.vector_store.chunks)} | "
                f"Prototypes: {len(self.prototype_manager.prototypes)} | "
                f"KG Edges: {len(self.temporal_kg.edges)}"
            )

    async def query(self, question: str, time_window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Full RAG query pipeline:
        trigger evaluation → temporal filtering → hybrid retrieval → KG lookup → generation.
        """
        ref = now_utc()
        logger.info(f"Query received: '{question}'")

        # 1. Retrieval trigger evaluation
        trigger = self.retrieval_trigger.evaluate(question, explicit_window_minutes=time_window_minutes)

        if not trigger.should_retrieve:
            return {
                "answer": trigger.reason,
                "query": question,
                "timestamp": to_iso_utc(ref),
                "trigger": {"should_retrieve": False, "reason": trigger.reason},
                "sources": [],
                "kg_facts": [],
            }

        # 2. Embed query
        query_embedding = self.embedder.embed_text(question)

        # 3. Dense vector search with temporal window
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=15,
            start_time=trigger.time_window_start,
            end_time=trigger.time_window_end,
        )

        # 4. Hybrid re-ranking (dense + sparse BM25 + recency)
        ranked = self.hybrid_searcher.rank(
            query=question,
            query_embedding=query_embedding,
            candidates=dense_results,
            reference_time=ref,
        )

        # 5. Knowledge graph temporal lookup
        kg_edges = self.temporal_kg.query_window(
            start_time=trigger.time_window_start,
            end_time=trigger.time_window_end,
        )

        # 6. Generate timestamped answer with citations
        result = self.generator.generate(
            query=question,
            retrieved_chunks=ranked[:8],
            kg_edges=kg_edges[:5],
            reference_time=ref,
        )

        result["trigger"] = {
            "should_retrieve": trigger.should_retrieve,
            "reason": trigger.reason,
            "confidence": trigger.confidence,
            "intent": trigger.detected_intent,
            "time_window_start": to_iso_utc(trigger.time_window_start) if trigger.time_window_start else None,
            "time_window_end": to_iso_utc(trigger.time_window_end) if trigger.time_window_end else None,
        }

        logger.info(
            f"Query answered. Sources: {len(result['sources'])}, "
            f"KG Facts: {len(result['kg_facts'])}, "
            f"Confidence: {result['confidence']:.3f}"
        )
        return result

    def stop_ingestion(self):
        """Gracefully stop the background ingestion loop."""
        self.stream_simulator.stop()
        if self._ingestion_task and not self._ingestion_task.done():
            self._ingestion_task.cancel()
        logger.info("Ingestion stopped.")

    def get_system_status(self) -> Dict[str, Any]:
        """Return full system health and diagnostics."""
        return {
            "project": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "timestamp": to_iso_utc(),
            "ingestion": {
                "is_running": self.stream_simulator.is_running,
                "total_chunks_ingested": self._chunks_ingested,
            },
            "vector_store": self.vector_store.get_stats(),
            "prototypes": self.prototype_manager.get_summary(),
            "temporal_kg": self.temporal_kg.get_summary(),
            "config": {
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "llm_provider": settings.LLM_PROVIDER,
                "max_memory_chunks": settings.MAX_MEMORY_CHUNKS,
                "max_prototypes": settings.MAX_ACTIVE_PROTOTYPES,
                "decay_half_life_sec": settings.TEMPORAL_DECAY_HALF_LIFE_SEC,
            },
        }

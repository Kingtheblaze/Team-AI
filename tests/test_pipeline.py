"""
ChronoStream RAG — Test Suite.
Tests core pipeline components end-to-end.
"""
import asyncio
import pytest
from datetime import timedelta

from app.config import settings
from app.ingestion.chunker import StreamChunker, StreamChunk
from app.ingestion.stream_simulator import StreamSimulator
from app.prototypes.decay import TemporalDecay
from app.prototypes.clustering import MicroCluster, cosine_similarity
from app.prototypes.prototype_manager import HierarchicalPrototypeManager
from app.temporal_kg.entity_extractor import TemporalEntityExtractor
from app.temporal_kg.graph import TemporalKnowledgeGraph
from app.retrieval.vector_store import EmbeddingEngine, TemporalVectorStore
from app.retrieval.trigger import RetrievalTrigger
from app.retrieval.hybrid_search import HybridTimeAwareSearcher
from app.generation.generator import AnswerGenerator
from app.utils.time_utils import now_utc, to_iso_utc, parse_time_window_to_delta, extract_temporal_bounds


# ── Time Utilities ──────────────────────────────────────────────────────

class TestTimeUtils:
    def test_now_utc_has_timezone(self):
        dt = now_utc()
        assert dt.tzinfo is not None

    def test_to_iso_utc_format(self):
        iso = to_iso_utc()
        assert iso.endswith("Z")
        assert "T" in iso

    def test_parse_time_window_minutes(self):
        delta = parse_time_window_to_delta("last 10 minutes")
        assert delta == timedelta(minutes=10)

    def test_parse_time_window_hours(self):
        delta = parse_time_window_to_delta("past 2 hours")
        assert delta == timedelta(hours=2)

    def test_parse_time_window_seconds(self):
        delta = parse_time_window_to_delta("last 30 seconds")
        assert delta == timedelta(seconds=30)

    def test_extract_temporal_bounds_with_window(self):
        start, end = extract_temporal_bounds("last 5 minutes")
        assert start is not None
        assert end is not None
        assert (end - start).total_seconds() == pytest.approx(300, abs=2)


# ── Chunker ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_single_short_chunk(self):
        chunker = StreamChunker()
        chunks = chunker.chunk_text("Hello world test message")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world test message"

    def test_chunk_has_timestamp(self):
        chunker = StreamChunker()
        chunks = chunker.chunk_text("Test data", source="test://src")
        assert chunks[0].timestamp is not None
        assert chunks[0].source == "test://src"

    def test_empty_text_returns_empty(self):
        chunker = StreamChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_long_text_splits(self):
        chunker = StreamChunker(max_chunk_chars=50)
        long_text = "word " * 100
        chunks = chunker.chunk_text(long_text.strip())
        assert len(chunks) > 1


# ── Stream Simulator ───────────────────────────────────────────────────

class TestStreamSimulator:
    @pytest.mark.asyncio
    async def test_generate_event(self):
        sim = StreamSimulator(interval_seconds=0.1)
        chunk = await sim.generate_event()
        assert isinstance(chunk, StreamChunk)
        assert chunk.text
        assert chunk.timestamp is not None


# ── Temporal Decay ──────────────────────────────────────────────────────

class TestTemporalDecay:
    def test_current_event_decay_is_one(self):
        decay = TemporalDecay(half_life_seconds=300)
        now = now_utc()
        factor = decay.decay_factor(now, now)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_half_life_event_decay_is_half(self):
        decay = TemporalDecay(half_life_seconds=300)
        now = now_utc()
        past = now - timedelta(seconds=300)
        factor = decay.decay_factor(past, now)
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_salience_increases_with_count(self):
        decay = TemporalDecay(half_life_seconds=300)
        now = now_utc()
        s1 = decay.compute_salience_score(count=1, last_updated=now, reference_time=now)
        s5 = decay.compute_salience_score(count=5, last_updated=now, reference_time=now)
        assert s5 > s1


# ── Prototype Manager ──────────────────────────────────────────────────

class TestPrototypeManager:
    def test_assimilate_creates_prototype(self):
        mgr = HierarchicalPrototypeManager(max_prototypes=10)
        embedding = [0.1] * settings.EMBEDDING_DIMENSION
        proto = mgr.assimilate_chunk(embedding, "Test event", "test://src")
        assert proto is not None
        assert len(mgr.prototypes) == 1

    def test_bounded_memory_eviction(self):
        mgr = HierarchicalPrototypeManager(max_prototypes=3, similarity_threshold=0.99)
        for i in range(10):
            emb = [0.0] * settings.EMBEDDING_DIMENSION
            emb[i % settings.EMBEDDING_DIMENSION] = 1.0
            mgr.assimilate_chunk(emb, f"Event {i}", f"src-{i}")
        assert len(mgr.prototypes) <= 3


# ── Entity Extractor ───────────────────────────────────────────────────

class TestEntityExtractor:
    def test_extract_known_entities(self):
        ext = TemporalEntityExtractor()
        entities = ext.extract_entities("Samsung semiconductor lab announces 1.4nm GAAFET wafer.")
        assert "Samsung" in entities
        assert "GAAFET" in entities

    def test_extract_triplets(self):
        ext = TemporalEntityExtractor()
        triplets = ext.extract_triplets("Samsung announces next-gen chip design.")
        assert len(triplets) >= 1


# ── Temporal Knowledge Graph ───────────────────────────────────────────

class TestTemporalKG:
    def test_add_and_query(self):
        kg = TemporalKnowledgeGraph()
        chunk = StreamChunk(
            id="test-1",
            text="Samsung announces GAAFET wafer production.",
            timestamp=now_utc(),
            source="test://src",
        )
        kg.add_chunk(chunk)
        assert len(kg.edges) >= 1

    def test_bounded_edge_count(self):
        kg = TemporalKnowledgeGraph(max_edges=5)
        for i in range(20):
            chunk = StreamChunk(
                id=f"test-{i}",
                text="Samsung announces new product line.",
                timestamp=now_utc(),
                source="test://src",
            )
            kg.add_chunk(chunk)
        assert len(kg.edges) <= 5


# ── Embedding Engine ───────────────────────────────────────────────────

class TestEmbeddingEngine:
    def test_embed_text_returns_vector(self):
        engine = EmbeddingEngine()
        vec = engine.embed_text("test sentence")
        assert len(vec) == settings.EMBEDDING_DIMENSION
        assert any(v != 0 for v in vec)


# ── Vector Store ───────────────────────────────────────────────────────

class TestVectorStore:
    def test_add_and_search(self):
        store = TemporalVectorStore(max_chunks=100)
        engine = EmbeddingEngine()
        chunk = StreamChunk(
            id="v-1", text="Hello world", timestamp=now_utc(), source="test://src"
        )
        emb = engine.embed_text(chunk.text)
        store.add_chunk(chunk, emb)
        results = store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0][0].id == "v-1"

    def test_bounded_capacity(self):
        store = TemporalVectorStore(max_chunks=5)
        engine = EmbeddingEngine()
        for i in range(20):
            chunk = StreamChunk(
                id=f"v-{i}", text=f"Event number {i}", timestamp=now_utc(), source="test"
            )
            store.add_chunk(chunk, engine.embed_text(chunk.text))
        assert len(store.chunks) <= 5


# ── Retrieval Trigger ──────────────────────────────────────────────────

class TestRetrievalTrigger:
    def test_temporal_query_triggers(self):
        trigger = RetrievalTrigger()
        decision = trigger.evaluate("What happened in the last 10 minutes?")
        assert decision.should_retrieve is True

    def test_chit_chat_does_not_trigger(self):
        trigger = RetrievalTrigger()
        decision = trigger.evaluate("Hello there!")
        assert decision.should_retrieve is False


# ── Hybrid Search ──────────────────────────────────────────────────────

class TestHybridSearch:
    def test_rank_returns_sorted(self):
        searcher = HybridTimeAwareSearcher()
        engine = EmbeddingEngine()
        q_emb = engine.embed_text("test")
        chunk1 = StreamChunk(id="h-1", text="test sentence one", timestamp=now_utc(), source="s")
        chunk2 = StreamChunk(id="h-2", text="unrelated data xyz", timestamp=now_utc(), source="s")
        candidates = [
            (chunk1, 0.9),
            (chunk2, 0.3),
        ]
        ranked = searcher.rank("test", q_emb, candidates)
        assert ranked[0][0].id == "h-1"


# ── Generator ──────────────────────────────────────────────────────────

class TestGenerator:
    def test_generate_with_empty_context(self):
        gen = AnswerGenerator(provider="fast_extractive")
        result = gen.generate("What happened?", retrieved_chunks=[])
        assert "No relevant" in result["answer"]
        assert result["sources"] == []

    def test_generate_with_context(self):
        gen = AnswerGenerator(provider="fast_extractive")
        chunk = StreamChunk(id="g-1", text="Server rebooted at 14:20", timestamp=now_utc(), source="test")
        result = gen.generate("What happened?", retrieved_chunks=[(chunk, 0.85, {"dense_score": 0.85, "sparse_score": 0.1, "recency_score": 0.9, "age_seconds": 5, "final_score": 0.8})])
        assert len(result["sources"]) == 1
        assert "timestamp" in result

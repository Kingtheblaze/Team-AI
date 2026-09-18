"""
Answer Generator with Timestamped Source Citations.
Fast extractive pipeline by default; pluggable LLM backends (Ollama, OpenAI, vLLM).
"""
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from app.config import settings
from app.ingestion.chunker import StreamChunk
from app.temporal_kg.graph import TemporalEdge
from app.utils.logger import get_logger
from app.utils.time_utils import to_iso_utc, now_utc

logger = get_logger("Generator")


class AnswerGenerator:
    """
    Produces timestamped, source-attributed answers from retrieved context.
    Default mode: fast extractive summarization (no external LLM dependency).
    Optional: route through Ollama/OpenAI for richer generative answers.
    """

    def __init__(self, provider: str = settings.LLM_PROVIDER):
        self.provider = provider
        self._llm_client = None
        if provider == "ollama":
            self._init_ollama()
        elif provider == "openai":
            self._init_openai()

    def _init_ollama(self):
        try:
            import httpx
            self._llm_client = httpx.Client(base_url=settings.OLLAMA_BASE_URL, timeout=30.0)
            logger.info(f"Ollama client initialized → {settings.OLLAMA_BASE_URL}")
        except Exception as e:
            logger.warning(f"Ollama init failed ({e}). Falling back to extractive mode.")
            self.provider = "fast_extractive"

    def _init_openai(self):
        try:
            from openai import OpenAI  # type: ignore
            self._llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized.")
        except Exception as e:
            logger.warning(f"OpenAI init failed ({e}). Falling back to extractive mode.")
            self.provider = "fast_extractive"

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Tuple[StreamChunk, float, dict]],
        kg_edges: Optional[List[TemporalEdge]] = None,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer with mandatory UTC timestamps and source citations.
        """
        ref = reference_time or now_utc()

        if not retrieved_chunks:
            return {
                "answer": "No relevant streaming context found for this query within the specified time window.",
                "query": query,
                "timestamp": to_iso_utc(ref),
                "sources": [],
                "kg_facts": [],
                "confidence": 0.0,
            }

        # Build context block from retrieved chunks
        context_lines = []
        sources = []
        for i, (chunk, score, breakdown) in enumerate(retrieved_chunks[:8]):
            context_lines.append(
                f"[{i+1}] [{to_iso_utc(chunk.timestamp)}] ({chunk.source}) {chunk.text}"
            )
            sources.append({
                "rank": i + 1,
                "chunk_id": chunk.id,
                "text": chunk.text,
                "timestamp": to_iso_utc(chunk.timestamp),
                "source": chunk.source,
                "relevance_score": round(score, 4),
                "score_breakdown": breakdown,
            })

        context_block = "\n".join(context_lines)

        # Build KG facts block
        kg_facts = []
        kg_block = ""
        if kg_edges:
            kg_lines = []
            for edge in kg_edges[:5]:
                fact_str = f"  • {edge.subject} → {edge.predicate} → {edge.object} [{to_iso_utc(edge.timestamp)}]"
                kg_lines.append(fact_str)
                kg_facts.append(edge.to_dict())
            kg_block = "\nKnowledge Graph Facts:\n" + "\n".join(kg_lines)

        # Route to appropriate generation backend
        if self.provider == "fast_extractive":
            answer = self._extractive_answer(query, context_block, kg_block, sources)
        elif self.provider == "ollama":
            answer = self._ollama_answer(query, context_block, kg_block)
        elif self.provider == "openai":
            answer = self._openai_answer(query, context_block, kg_block)
        else:
            answer = self._extractive_answer(query, context_block, kg_block, sources)

        avg_score = sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0.0

        return {
            "answer": answer,
            "query": query,
            "timestamp": to_iso_utc(ref),
            "sources": sources,
            "kg_facts": kg_facts,
            "confidence": round(avg_score, 4),
            "provider": self.provider,
        }

    def _extractive_answer(
        self,
        query: str,
        context_block: str,
        kg_block: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Fast zero-dependency extractive answer construction.
        Selects the most relevant sentences from retrieved context and
        synthesizes them into a coherent timestamped narrative.
        """
        header = f"Based on live streaming data (as of {to_iso_utc()}):\n\n"
        body_parts = []
        for s in sources[:5]:
            body_parts.append(
                f"• [{s['timestamp']}] {s['text']} (source: {s['source']}, relevance: {s['relevance_score']:.2f})"
            )
        body = "\n".join(body_parts)
        footer = ""
        if kg_block:
            footer = f"\n\nRelated knowledge graph relationships:\n{kg_block}"
        return header + body + footer

    def _ollama_answer(self, query: str, context_block: str, kg_block: str) -> str:
        """Generate answer via local Ollama instance."""
        prompt = self._build_llm_prompt(query, context_block, kg_block)
        try:
            resp = self._llm_client.post(
                "/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "Generation failed.")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"[Ollama error: {e}] Falling back to raw context:\n{context_block}"

    def _openai_answer(self, query: str, context_block: str, kg_block: str) -> str:
        """Generate answer via OpenAI API."""
        prompt = self._build_llm_prompt(query, context_block, kg_block)
        try:
            resp = self._llm_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a real-time streaming RAG assistant. Answer using ONLY the provided timestamped context. Always cite UTC timestamps and sources."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"[OpenAI error: {e}] Falling back to raw context:\n{context_block}"

    def _build_llm_prompt(self, query: str, context_block: str, kg_block: str) -> str:
        return (
            f"You are ChronoStream RAG, a real-time streaming retrieval-augmented assistant.\n"
            f"Answer the user's question using ONLY the timestamped context below.\n"
            f"Every claim MUST include a UTC timestamp and source attribution.\n"
            f"If the context is insufficient, say so.\n\n"
            f"--- LIVE STREAMING CONTEXT ---\n{context_block}\n{kg_block}\n"
            f"--- END CONTEXT ---\n\n"
            f"User Question: {query}\n\n"
            f"Answer (with timestamps and sources):"
        )

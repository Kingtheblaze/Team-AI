"""
ChronoStream RAG - FastAPI REST API.
Exposes query, ingestion control, and system status endpoints.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import ChronoStreamPipeline
from app.utils.logger import get_logger

logger = get_logger("API")

pipeline: Optional[ChronoStreamPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize pipeline and start ingestion on startup."""
    global pipeline
    logger.info("Starting ChronoStream RAG API...")
    pipeline = ChronoStreamPipeline()
    await pipeline.start_ingestion()
    logger.info("API ready. Ingestion running.")
    yield
    logger.info("Shutting down...")
    if pipeline:
        pipeline.stop_ingestion()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Adaptive Temporal Streaming RAG API. "
        "Continuously ingests live data streams and answers time-sensitive queries "
        "with precise UTC timestamps and source citations."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question to answer from the live stream.")
    time_window_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Optional: restrict retrieval to last N minutes.")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest into the streaming pipeline.")
    source: str = Field("stream://manual-input", description="Source identifier for provenance.")


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs": "/docs",
    }


@app.post("/query", tags=["RAG"])
async def query_endpoint(req: QueryRequest):
    """
    Submit a natural-language query against the live streaming temporal index.
    Returns a timestamped answer with source citations and KG facts.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    result = await pipeline.query(
        question=req.question,
        time_window_minutes=req.time_window_minutes,
    )
    return result


@app.post("/ingest", tags=["Ingestion"])
async def ingest_text(req: IngestTextRequest):
    """Manually ingest a text snippet into the streaming pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    from app.ingestion.chunker import StreamChunker
    from app.utils.time_utils import now_utc

    chunker = StreamChunker()
    chunks = chunker.chunk_text(text=req.text, source=req.source, timestamp=now_utc())
    for chunk in chunks:
        await pipeline.ingest_chunk(chunk)
    return {"status": "ingested", "chunks_created": len(chunks)}


@app.post("/ingestion/start", tags=["Ingestion"])
async def start_ingestion():
    """Start the simulated continuous stream ingestion."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    await pipeline.start_ingestion()
    return {"status": "ingestion_started"}


@app.post("/ingestion/stop", tags=["Ingestion"])
async def stop_ingestion():
    """Stop the background ingestion loop."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    pipeline.stop_ingestion()
    return {"status": "ingestion_stopped"}


@app.get("/status", tags=["Diagnostics"])
async def system_status():
    """Return full system health, memory stats, and prototype diagnostics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return pipeline.get_system_status()


@app.get("/prototypes", tags=["Diagnostics"])
async def get_prototypes():
    """Return current hierarchical adaptive prototype state."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return pipeline.prototype_manager.get_summary()


@app.get("/kg", tags=["Diagnostics"])
async def get_knowledge_graph():
    """Return temporal knowledge graph snapshot."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return pipeline.temporal_kg.get_summary()

# ChronoStream RAG

**Adaptive Temporal Streaming Retrieval-Augmented Generation**

> Samsung PRISM Generative AI Hackathon – 3rd Edition (2026–27)
> **Theme:** Streaming Live RAG
> **Team:** Team AI
> **College:** MS Ramaiah Institute of Technology
> **Member:** Sambhav Heda

---

## Overview

Traditional RAG systems rely on **static** knowledge bases. On live data streams (news, audio, sensors, markets) the retrieved context becomes stale within minutes, leading to hallucinations and outdated answers.

**ChronoStream RAG** is a streaming-native RAG system that:

- Continuously ingests live data streams
- Maintains an always-fresh, time-aware index
- Answers natural-language queries with precise temporal grounding and source attribution
- Keeps memory bounded while preserving semantic coverage
- Supports low perceived latency through learned retrieval triggering and prefetching

---

## Problem Statement

Build a system that can answer questions such as:

- "What were the main topics discussed in the last 10 minutes?"
- "Any critical events since 14:20 UTC?"
- "Summarize the latest developments from the live feed."

…while guaranteeing that the retrieved context is only seconds old, not hours old.

---

## Key Gaps We Address

| Gap | How ChronoStream Handles It |
|-----|----------------------------|
| Memory vs Semantic Coverage | Hierarchical Adaptive Prototypes (micro-clusters + global heavy-hitters with temporal decay) |
| Persistent Staleness | Event-driven incremental upserts + continuous temporal Knowledge Graph |
| High latency of heavy models | Lightweight extractors + learned retrieval trigger + parallel prefetch |
| Weak temporal reasoning | Hybrid dense + sparse + temporal metadata retrieval with recency bias |
| Poor source attribution | Every answer carries precise UTC timestamps and source chunks |

---

## Architecture

```
Live Streams (Text / News / Sensors / Audio)
          │
          ▼
┌─────────────────────────────────┐
│  Continuous Stream Ingestion    │  ← chunking + embedding (incremental)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Hierarchical Adaptive           │  ← online micro-clusters + heavy-hitters
│ Prototypes + Temporal KG        │     + exponential temporal decay
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Retrieval Trigger Engine        │  ← rule-based intent + temporal detection
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Hybrid Time-Aware Retriever     │  ← dense + sparse BM25 + recency bias
│ + Answer Generator              │     + timestamped citations
└─────────────────────────────────┘
          │
          ▼
     Answer with UTC timestamps + sources
```

---

## Features

- [x] Continuous simulated stream ingestion (async event loop)
- [x] Incremental vector + temporal metadata updates (never full rebuild)
- [x] Hierarchical adaptive prototype memory management (bounded, decayed)
- [x] Temporal Knowledge Graph with entity-relation extraction
- [x] Hybrid retrieval: dense + sparse BM25 + temporal recency scoring
- [x] Retrieval trigger engine (separates temporal queries from chit-chat)
- [x] Time-window queries ("last N minutes", "since HH:MM UTC")
- [x] Precise source attribution with UTC timestamps on every answer
- [x] Bounded memory guarantees (configurable max chunks + max prototypes)
- [x] FastAPI REST API with full OpenAPI docs
- [x] Streamlit interactive demo dashboard
- [x] Docker + docker-compose deployment
- [x] Comprehensive test suite (pytest)
- [ ] Learned retrieval trigger model (planned v2)
- [ ] Multi-modal support: audio via Faster-Whisper (scaffolded, optional)

---

## Tech Stack

| Component              | Choice                                      |
|------------------------|---------------------------------------------|
| Language               | Python 3.11+                                |
| API Framework          | FastAPI + Uvicorn                           |
| Embeddings             | sentence-transformers (all-MiniLM-L6-v2)    |
| Vector Store           | In-memory temporal store (Chroma/Qdrant pluggable) |
| Streaming              | asyncio event loop + simulated stream       |
| Temporal Graph         | In-memory KG with bounded edge retention    |
| LLM / Generation       | Fast extractive (default) / Ollama / OpenAI |
| ASR (optional)         | Faster-Whisper (graceful fallback)          |
| Demo UI                | Streamlit                                   |
| Packaging              | Docker + docker-compose                    |

---

## Project Structure

```
Team-AI/
├── README.md                          # This file
├── AI_DISCLOSURE.md                   # AI tool usage transparency
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
├── docker-compose.yml                 # Multi-container deployment
├── Dockerfile                         # Container build
├── main.py                            # Entry point (FastAPI server)
├── app/
│   ├── __init__.py
│   ├── api.py                         # FastAPI REST endpoints
│   ├── config.py                      # Pydantic settings & configuration
│   ├── pipeline.py                    # Core orchestration engine
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py                 # StreamChunk model + text chunker
│   │   ├── stream_simulator.py        # Continuous simulated event stream
│   │   └── whisper_transcriber.py     # Audio transcription (optional)
│   ├── prototypes/
│   │   ├── __init__.py
│   │   ├── clustering.py              # MicroCluster online learning
│   │   ├── decay.py                   # Exponential temporal decay
│   │   └── prototype_manager.py       # Hierarchical adaptive memory
│   ├── temporal_kg/
│   │   ├── __init__.py
│   │   ├── entity_extractor.py        # Rule-based entity/relation extraction
│   │   └── graph.py                   # In-memory temporal knowledge graph
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py            # Embedding engine + temporal vector DB
│   │   ├── trigger.py                 # Retrieval trigger engine
│   │   └── hybrid_search.py           # Dense + BM25 + recency re-ranker
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py              # Timestamped answer generator
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Structured logging
│       └── time_utils.py             # UTC time parsing & window extraction
├── demo/
│   └── streamlit_app.py              # Interactive Streamlit dashboard
├── scripts/
│   └── smoke_test.py                 # End-to-end validation script
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py             # Comprehensive pytest suite
└── MSRamaiah_TeamAi_4StreamingRAG_Submission (1).pptx  # Presentation
```

---

## Quick Start

### Option A: Local Python (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/Kingtheblaze/Team-AI.git
cd Team-AI

# 2. Create and activate virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
# (Edit .env if you want to use Ollama or OpenAI instead of the default extractive mode)

# 5. Start the API server (auto-starts continuous stream ingestion)
python main.py
# API will be live at http://localhost:8000
# Swagger docs at http://localhost:8000/docs

# 6. In a separate terminal, launch the Streamlit demo
streamlit run demo/streamlit_app.py
# Dashboard will open at http://localhost:8501
```

### Option B: Docker Compose (One Command)

```bash
# 1. Clone and enter the repo
git clone https://github.com/Kingtheblaze/Team-AI.git
cd Team-AI

# 2. Copy environment config
cp .env.example .env

# 3. Build and run
docker compose up --build
# API → http://localhost:8000
# Demo → http://localhost:8501
```

### Option C: Quick Smoke Test (No Server Needed)

```bash
# After installing dependencies (steps 1-4 from Option A):
python scripts/smoke_test.py
```

This runs the full pipeline end-to-end: starts ingestion → waits for data → queries → prints timestamped answer.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/query` | Submit a temporal RAG query |
| `POST` | `/ingest` | Manually inject text into stream |
| `POST` | `/ingestion/start` | Start continuous ingestion |
| `POST` | `/ingestion/stop` | Stop continuous ingestion |
| `GET` | `/status` | Full system diagnostics |
| `GET` | `/prototypes` | View active hierarchical prototypes |
| `GET` | `/kg` | View temporal knowledge graph |
| `GET` | `/docs` | Interactive Swagger API docs |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happened in the last 5 minutes?", "time_window_minutes": 5}'
```

### Example Response

```json
{
  "answer": "Based on live streaming data (as of 2026-09-18T16:35:00Z):\n\n• [2026-09-18T16:34:52Z] SOC Incident #4082: Anomalous outbound TLS traffic detected on port 8443 from gateway-04. (source: stream://sec-ops, relevance: 0.92)\n• [2026-09-18T16:34:50Z] Samsung semiconductor lab announces 1.4nm GAAFET test wafer tapeout. (source: stream://robotics-feed, relevance: 0.87)",
  "timestamp": "2026-09-18T16:35:00Z",
  "sources": [...],
  "kg_facts": [...],
  "confidence": 0.895
}
```

---

## Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio

# Run the full test suite
pytest tests/ -v
```

---

## Configuration

All settings are configurable via environment variables or `.env` file. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `fast_extractive` | `fast_extractive`, `ollama`, or `openai` |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model name |
| `MAX_MEMORY_CHUNKS` | `500` | Max vectors in memory (bounded) |
| `MAX_ACTIVE_PROTOTYPES` | `50` | Max adaptive prototype clusters |
| `TEMPORAL_DECAY_HALF_LIFE_SEC` | `300` | Decay half-life in seconds |
| `STREAM_INTERVAL_SECONDS` | `2.0` | Simulated stream event interval |
| `WEIGHT_DENSE` | `0.50` | Dense similarity weight in hybrid score |
| `WEIGHT_SPARSE` | `0.25` | BM25 sparse weight |
| `WEIGHT_RECENCY` | `0.25` | Temporal recency weight |

---

## Demo Flow

1. **Start the system** — API server begins continuous stream ingestion automatically
2. **Open the Streamlit dashboard** — see live metrics, prototype states, and KG edges
3. **Ask a time-sensitive question** — e.g., "What security incidents happened in the last 5 minutes?"
4. **Receive a timestamped answer** — with source citations, score breakdowns, and KG facts
5. **Inject custom events** — manually ingest text and immediately query it
6. **Observe bounded memory** — watch prototypes merge/evict as memory limits are enforced

---

## Team

| Name | Role | College |
|------|------|---------|
| Sambhav Heda | Sole Member / Lead | MS Ramaiah Institute of Technology |

---

## Acknowledgments

- Samsung PRISM Generative AI Hackathon (3rd Edition) — Language AI Team & PRISM Team, Samsung R&D Institute India
- Research inspiration from recent Streaming RAG literature (Zhu 2025, Sankaradas et al., Stream RAG, RisingWave, etc.)

---

## License

This project is developed for the Samsung PRISM Generative AI Hackathon 2026–27.
License will be finalized after the event (currently all rights reserved by the author).

---

**Status:** Fully implemented and runnable
**Last updated:** September 2026

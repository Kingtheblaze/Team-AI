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

- “What were the main topics discussed in the last 10 minutes?”
- “Any critical events since 14:20 UTC?”
- “Summarize the latest developments from the live feed.”

…while guaranteeing that the retrieved context is only seconds old, not hours old.

---

## Key Gaps We Address

| Gap | How ChronoStream Handles It |
|-----|-----------------------------|
| Memory vs Semantic Coverage | Hierarchical Adaptive Prototypes (micro-clusters + global heavy-hitters with temporal decay) |
| Persistent Staleness | Event-driven incremental upserts + continuous temporal Knowledge Graph |
| High latency of heavy models | Lightweight extractors + learned retrieval trigger + parallel prefetch |
| Weak temporal reasoning | Hybrid dense + sparse + temporal metadata retrieval with recency bias |
| Edge / Privacy | On-device index option + selective cloud offload |
| Poor source attribution | Every answer carries precise UTC timestamps and source chunks |

---

## Architecture (High Level)

```
Live Streams (Audio / News / Sensors / Logs)
          │
          ▼
┌─────────────────────────────┐
│  Lightweight Ingestion      │  ← chunking + quantized embeddings
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Hierarchical Adaptive       │  ← micro-clusters + heavy-hitters
│ Prototypes + Temporal KG    │     + event prioritization
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Learned Retrieval Trigger   │  ← decides when & what to retrieve
│ + Parallel Prefetch         │
└─────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Hybrid Time-Aware Retriever │  ← dense + sparse + temporal filters
│ + LLM Generation            │     + timestamped citations
└─────────────────────────────┘
          │
          ▼
     Answer with sources
```

---

## Features (Planned / In Progress)

- [x] Continuous stream ingestion
- [x] Incremental vector + temporal metadata updates
- [x] Hierarchical prototype memory management
- [x] Time-window queries (“last N minutes”)
- [x] Precise source attribution with UTC timestamps
- [ ] Learned retrieval trigger model
- [ ] On-device / edge mode
- [ ] Multi-modal support (audio + text + structured)
- [ ] Full evaluation suite on temporal QA

---

## Tech Stack

| Component              | Choice                                      |
|------------------------|---------------------------------------------|
| Language               | Python 3.11+                                |
| Orchestration          | LangChain / LlamaIndex + FastAPI            |
| Streaming              | asyncio / Redis Streams / Kafka (optional)  |
| Embeddings             | sentence-transformers (quantized)           |
| Vector Store           | Qdrant or Chroma (with temporal metadata)   |
| ASR (live audio)       | Faster-Whisper                              |
| LLM                    | Llama-3.x / Qwen2.5 / Gemma via vLLM/Ollama |
| Temporal Graph         | NetworkX / lightweight in-memory KG         |
| UI / Demo              | Streamlit or Gradio                         |
| Packaging              | Docker                                      |

---

## Quick Start (Coming Soon)

```bash
# Clone the repository
git clone https://github.com/<your-username>/ChronoStream-RAG.git
cd ChronoStream-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Pull models
# python scripts/download_models.py

# Start the system
python main.py
# or
docker compose up
```

Detailed setup instructions will be added once the core code is committed.

---

## Demo

A ≤ 5-minute demo video will be linked here once recorded.

**Demo flow:**
1. Start a live stream (audio / news / simulated sensors)
2. System continuously updates the temporal index in the background
3. Ask a time-sensitive question
4. Receive an answer with precise timestamps and source citations
5. Side-by-side comparison with static RAG (to show freshness advantage)

---

## Project Structure (Planned)

```
ChronoStream-RAG/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── main.py
├── src/
│   ├── ingestion/
│   ├── prototypes/
│   ├── temporal_kg/
│   ├── retrieval/
│   ├── trigger/
│   └── generation/
├── configs/
├── scripts/
├── notebooks/
└── demo/
```

---

## Team

| Name            | Role                  | College                          |
|-----------------|-----------------------|----------------------------------|
| Sambhav Heda    | Sole Member / Lead    | MS Ramaiah Institute of Technology |

---

## Acknowledgments

- Samsung PRISM Generative AI Hackathon (3rd Edition) – Language AI Team & PRISM Team, Samsung R&D Institute India
- Research inspiration from recent Streaming RAG literature (Zhu 2025, Sankaradas et al., Stream RAG, RisingWave, etc.)

---

## License

This project is developed for the Samsung PRISM Generative AI Hackathon 2026–27.  
License will be finalized after the event (currently all rights reserved by the author).

---

**Status:** Prototype under active development  
**Last updated:** September 2026

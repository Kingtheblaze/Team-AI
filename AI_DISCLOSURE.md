# AI Disclosure — ChronoStream RAG

> **Samsung PRISM Generative AI Hackathon — 3rd Edition (2026–27)**
> **Team:** Team AI | MS Ramaiah Institute of Technology
> **Member:** Sambhav Heda

---

## AI Tools Used During Development

This document provides full transparency on the use of AI-assisted tools during the development of ChronoStream RAG, as required by the hackathon submission guidelines.

### 1. Code Generation & Architecture Assistance

| Tool | Purpose | Scope of Use |
|------|---------|--------------|
| Google Gemini / Antigravity IDE | Code scaffolding, module implementation, test writing | Used to accelerate boilerplate generation and translate architectural design into working Python modules. All generated code was reviewed, validated, and integrated by the team member. |

### 2. What AI Helped With

- **Project scaffolding**: Initial directory structure and module stubs.
- **Boilerplate code**: FastAPI endpoint wiring, Pydantic model definitions, Docker configuration.
- **Implementation acceleration**: Translating the designed architecture (hierarchical prototypes, temporal decay, hybrid retrieval) into typed Python code.
- **Test suite generation**: Creating comprehensive pytest test cases for each subsystem.
- **Documentation**: README structure and setup instructions.

### 3. What Was Done Entirely by the Team

- **Core system architecture and design**: The hierarchical adaptive prototype concept, temporal knowledge graph design, hybrid time-aware retrieval formula, and retrieval trigger strategy were entirely conceived and designed by the team.
- **Algorithm design**: Temporal decay functions, micro-cluster online learning, BM25+dense+recency hybrid scoring formula, and bounded memory eviction policies.
- **Problem identification**: Identifying the staleness gap in traditional RAG systems on live data streams.
- **Research**: Literature review of streaming RAG approaches, temporal reasoning in retrieval systems, and online clustering techniques.
- **Integration decisions**: Choice of sentence-transformers, FastAPI, Streamlit, and the specific architectural layering.
- **Testing and validation**: Manual end-to-end testing, verifying correctness of temporal queries, and validating bounded memory guarantees.
- **Presentation**: All presentation materials, demo flow design, and narrative.

### 4. AI Usage Philosophy

AI tools were used as **productivity accelerators**, not as idea generators. The intellectual contribution — problem framing, architecture, algorithm design, and system evaluation — is entirely the work of the team member. AI assistance was limited to translating well-defined designs into code faster than manual typing.

### 5. Reproducibility

All source code is available in the repository. The system can be fully reproduced by following the README instructions without any AI tool dependency.

---

**Signed:** Sambhav Heda
**Date:** September 2026

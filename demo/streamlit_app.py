"""
ChronoStream RAG — Streamlit Demo Dashboard.
Real-time interactive demo showing continuous ingestion, temporal querying,
prototype memory, and knowledge graph visualization.
"""
import streamlit as st
import requests
import json
import time
from datetime import datetime

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoStream RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"


def safe_api_call(method: str, endpoint: str, **kwargs):
    """Make API call with error handling."""
    try:
        resp = getattr(requests, method)(f"{API_BASE}{endpoint}", timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to ChronoStream API. Make sure `python main.py` is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: -10px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .source-card {
        background: #1e293b;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.92rem;
    }

    .source-meta {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 6px;
    }

    .answer-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px;
        line-height: 1.7;
        font-size: 1rem;
    }

    .kg-fact {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-live {
        background: #065f46;
        color: #34d399;
    }

    .status-stopped {
        background: #7f1d1d;
        color: #fca5a5;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚡ ChronoStream RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Adaptive Temporal Streaming Retrieval-Augmented Generation — Samsung PRISM Hackathon</p>', unsafe_allow_html=True)
st.markdown("---")


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Stream", use_container_width=True):
            safe_api_call("post", "/ingestion/start")
            st.success("Ingestion started")
    with col2:
        if st.button("⏹️ Stop Stream", use_container_width=True):
            safe_api_call("post", "/ingestion/stop")
            st.warning("Ingestion stopped")

    st.markdown("---")
    st.markdown("### ⏱️ Time Window")
    time_window = st.slider("Restrict to last N minutes", 1, 60, 10)

    st.markdown("---")
    st.markdown("### 📊 System Status")
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state["refresh"] = True

    status = safe_api_call("get", "/status")
    if status:
        ing = status.get("ingestion", {})
        vs = status.get("vector_store", {})
        proto = status.get("prototypes", {})
        kg = status.get("temporal_kg", {})

        is_live = ing.get("is_running", False)
        badge_class = "status-live" if is_live else "status-stopped"
        badge_text = "● LIVE" if is_live else "● STOPPED"
        st.markdown(f'<span class="status-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

        st.metric("Chunks Ingested", ing.get("total_chunks_ingested", 0))
        st.metric("Vectors Stored", vs.get("total_stored_chunks", 0))
        st.metric("Active Prototypes", proto.get("total_active_prototypes", 0))
        st.metric("KG Edges", kg.get("total_edges", 0))

    st.markdown("---")
    st.markdown("### 📝 Manual Ingest")
    manual_text = st.text_area("Inject text into stream:", height=100, placeholder="Type a custom event...")
    if st.button("📥 Ingest", use_container_width=True) and manual_text:
        res = safe_api_call("post", "/ingest", json={"text": manual_text, "source": "stream://manual-input"})
        if res:
            st.success(f"Ingested {res.get('chunks_created', 0)} chunk(s)")


# ── Main Area: Query ────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Time-Sensitive Question")

query_col, btn_col = st.columns([5, 1])
with query_col:
    user_query = st.text_input(
        "Question",
        placeholder="e.g., What happened in the last 5 minutes?",
        label_visibility="collapsed",
    )
with btn_col:
    query_btn = st.button("Ask ⚡", use_container_width=True, type="primary")

if query_btn and user_query:
    with st.spinner("Retrieving from live temporal index..."):
        result = safe_api_call("post", "/query", json={
            "question": user_query,
            "time_window_minutes": time_window,
        })

    if result:
        st.markdown("---")

        # ── Answer Box ──
        st.markdown("#### 💡 Answer")
        answer_text = result.get("answer", "No answer generated.")
        st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.caption(f"🕐 Generated at: `{result.get('timestamp', 'N/A')}`")
        with meta_col2:
            st.caption(f"📊 Confidence: `{result.get('confidence', 0):.3f}`")
        with meta_col3:
            trigger = result.get("trigger", {})
            st.caption(f"🎯 Intent: `{trigger.get('intent', 'N/A')}`")

        # ── Sources ──
        sources = result.get("sources", [])
        if sources:
            st.markdown("#### 📄 Sources & Citations")
            for src in sources:
                breakdown = src.get("score_breakdown", {})
                st.markdown(f"""
                <div class="source-card">
                    <strong>[#{src['rank']}]</strong> {src['text']}
                    <div class="source-meta">
                        🕐 {src['timestamp']} &nbsp;|&nbsp;
                        📡 {src['source']} &nbsp;|&nbsp;
                        🎯 Dense: {breakdown.get('dense_score', 0):.3f} &nbsp;
                        📝 Sparse: {breakdown.get('sparse_score', 0):.3f} &nbsp;
                        ⏱️ Recency: {breakdown.get('recency_score', 0):.3f} &nbsp;
                        ⚡ Final: {breakdown.get('final_score', 0):.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── KG Facts ──
        kg_facts = result.get("kg_facts", [])
        if kg_facts:
            st.markdown("#### 🧠 Knowledge Graph Facts")
            for fact in kg_facts:
                st.markdown(
                    f'<div class="kg-fact">'
                    f'{fact["subject"]} → <strong>{fact["predicate"]}</strong> → {fact["object"]} '
                    f'<span style="color:#64748b">| {fact["timestamp"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Tabs: Prototypes & KG ──────────────────────────────────────────────
st.markdown("---")
tab_proto, tab_kg, tab_stream = st.tabs(["🧩 Active Prototypes", "🕸️ Knowledge Graph", "📡 Stream Log"])

with tab_proto:
    proto_data = safe_api_call("get", "/prototypes")
    if proto_data:
        prototypes = proto_data.get("prototypes", [])
        if prototypes:
            for p in prototypes[:15]:
                with st.expander(f"🔹 {p['label']} (count: {p['count']}, salience: {p['salience_score']:.3f})"):
                    st.json(p)
        else:
            st.info("No active prototypes yet. Wait for stream ingestion to begin.")

with tab_kg:
    kg_data = safe_api_call("get", "/kg")
    if kg_data:
        edges = kg_data.get("recent_edges", [])
        if edges:
            for e in edges:
                st.markdown(
                    f'<div class="kg-fact">'
                    f'{e["subject"]} → <strong>{e["predicate"]}</strong> → {e["object"]} '
                    f'<span style="color:#64748b">| {e["timestamp"]} | {e["source_url"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No knowledge graph edges yet.")

with tab_stream:
    if status:
        ing = status.get("ingestion", {})
        st.markdown(f"**Total ingested:** {ing.get('total_chunks_ingested', 0)} chunks")
        vs = status.get("vector_store", {})
        st.markdown(f"**Vector store:** {vs.get('total_stored_chunks', 0)} / {vs.get('max_capacity', 'N/A')}")
        st.markdown(f"**Retention window:** {vs.get('retention_window_minutes', 'N/A')} minutes")


# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.85rem;">'
    'ChronoStream RAG — Samsung PRISM Gen AI Hackathon 2026 — Team AI — Sambhav Heda — MS Ramaiah Institute of Technology'
    '</p>',
    unsafe_allow_html=True,
)

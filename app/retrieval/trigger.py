"""
Retrieval Trigger Engine.
Determines whether an incoming query requires real-time streaming retrieval,
and extracts temporal parameters dynamically.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import re

from app.utils.time_utils import extract_temporal_bounds, now_utc


@dataclass
class TriggerDecision:
    should_retrieve: bool
    reason: str
    confidence: float
    time_window_start: Optional[datetime]
    time_window_end: Optional[datetime]
    detected_intent: str


class RetrievalTrigger:
    """
    Decides when and how to trigger streaming retrieval.
    Prevents redundant retrieval operations on static/general chit-chat queries
    and automatically identifies time-sensitive information requests.
    """

    TEMPORAL_TRIGGERS = [
        "last", "latest", "recent", "past", "since", "current", "now",
        "happened", "happening", "today", "status", "alert", "update",
        "incident", "telemetry", "breaking", "event", "new", "live"
    ]

    STATIC_PATTERNS = [
        r"^(hello|hi|hey|greetings|how are you)",
        r"^what is the capital of",
        r"^who wrote\b",
    ]

    def evaluate(self, query: str, explicit_window_minutes: Optional[int] = None) -> TriggerDecision:
        query_clean = query.strip().lower()

        # Check for obvious non-retrieval static chit-chat
        for pat in self.STATIC_PATTERNS:
            if re.search(pat, query_clean):
                return TriggerDecision(
                    should_retrieve=False,
                    reason="Static conversational query detected; no streaming context required.",
                    confidence=0.95,
                    time_window_start=None,
                    time_window_end=None,
                    detected_intent="chit_chat",
                )

        # Check for temporal triggers
        has_temporal_marker = any(re.search(rf"\b{word}\b", query_clean) for word in self.TEMPORAL_TRIGGERS)

        # Extract explicit or implicit temporal bounds
        start_t, end_t = extract_temporal_bounds(query, default_window_minutes=explicit_window_minutes)

        if has_temporal_marker or start_t is not None or explicit_window_minutes is not None:
            return TriggerDecision(
                should_retrieve=True,
                reason="Temporal marker or explicit time-window detected.",
                confidence=0.98,
                time_window_start=start_t,
                time_window_end=end_t or now_utc(),
                detected_intent="streaming_temporal_query",
            )

        # Default fallback: trigger retrieval for domain questions
        return TriggerDecision(
            should_retrieve=True,
            reason="General knowledge query against streaming live index.",
            confidence=0.80,
            time_window_start=start_t,
            time_window_end=end_t,
            detected_intent="standard_retrieval",
        )

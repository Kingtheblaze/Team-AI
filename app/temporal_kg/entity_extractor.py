"""
Rule-based & Regex Entity and Relationship Extraction for Streaming Ingestion.
Fast, deterministic, low-latency, and zero external API dependency.
"""
import re
from typing import List, Tuple


class TemporalEntityExtractor:
    """
    Extracts Subject-Predicate-Object triplets and named entities from streaming text chunks.
    Designed for real-time sub-millisecond execution without blocking the event loop.
    """

    KNOWN_ENTITIES = [
        "Samsung", "GAAFET", "Pyeongtaek", "NPU", "SOC", "gateway-04",
        "ETF", "Federal Reserve", "HBM4", "Dr. Kim", "Streaming RAG",
        "Prototypes", "us-east-gpu-09", "H100", "Incident #4082"
    ]

    RELATION_PATTERNS = [
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>announces|unveils|releases|launches)\s+(?P<obj>[\w\-\s\.]+)"), "ANNOUNCES"),
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>detected on|observed in)\s+(?P<obj>[\w\-\s\.]+)"), "DETECTED_AT"),
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>isolates|resolves|restores)\s+(?P<obj>[\w\-\s\.]+)"), "ACTION_ON"),
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>scaled up|increased|upgraded)\s+(?P<obj>[\w\-\s\.]+)"), "SCALED"),
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>reaches|achieved)\s+(?P<obj>[\w\-\s\.]+)"), "ACHIEVED"),
        (re.compile(r"(?P<sub>[A-Z][\w\-\s]+?)\s+(?P<pred>reports|notes|highlights)\s+(?P<obj>[\w\-\s\.]+)"), "REPORTS"),
    ]

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract (subject, predicate, object) triplets from text."""
        triplets: List[Tuple[str, str, str]] = []

        for pattern, rel_type in self.RELATION_PATTERNS:
            match = pattern.search(text)
            if match:
                sub = match.group("sub").strip()
                pred = match.group("pred").strip()
                obj = match.group("obj").strip()
                triplets.append((sub, pred, obj))

        # If no regex matched, synthesize fallback from known entities
        if not triplets:
            found_entities = [e for e in self.KNOWN_ENTITIES if e.lower() in text.lower()]
            if len(found_entities) >= 2:
                triplets.append((found_entities[0], "associated_with", found_entities[1]))
            elif len(found_entities) == 1:
                triplets.append((found_entities[0], "mentioned_in", "live_stream"))

        return triplets

    def extract_entities(self, text: str) -> List[str]:
        """Extract all prominent entity mentions from text."""
        entities = set()
        for e in self.KNOWN_ENTITIES:
            if re.search(rf"\b{re.escape(e)}\b", text, re.IGNORECASE):
                entities.add(e)

        # Also extract capitalized acronyms or hyphenated codes (e.g. HBM4, TLS)
        codes = re.findall(r"\b[A-Z]{2,}[0-9\-]*\b", text)
        for code in codes:
            if len(code) > 1 and code not in ["SOC", "TLS", "NPU", "ETF", "UTC"]:
                entities.add(code)

        return list(entities)

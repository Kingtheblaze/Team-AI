"""
Continuous Stream Simulator for ChronoStream RAG.
Generates realistic real-time events across tech, finance, incident response, and live audio transcripts.
"""
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional
import random

from app.ingestion.chunker import StreamChunk, StreamChunker
from app.utils.logger import get_logger
from app.utils.time_utils import now_utc

logger = get_logger("StreamSimulator")

SAMPLE_STREAM_EVENTS = [
    # Topic 1: Autonomous Systems & Hardware
    {"category": "tech", "source": "stream://robotics-feed", "text": "Samsung semiconductor lab announces 1.4nm GAAFET test wafer tapeout in Pyeongtaek campus."},
    {"category": "tech", "source": "stream://robotics-feed", "text": "Telemetry alert: Autonomous delivery drone fleet Alpha reached 99.4% mission completion in rain tests."},
    {"category": "tech", "source": "stream://robotics-feed", "text": "Edge NPU driver update 2.14 applied across robotic clusters, lowering inference latency to 1.8ms."},

    # Topic 2: Cyber Incident & Infrastructure
    {"category": "incident", "source": "stream://sec-ops", "text": "SOC Incident #4082: Anomalous outbound TLS traffic detected on port 8443 from gateway-04."},
    {"category": "incident", "source": "stream://sec-ops", "text": "Security Operations Center isolates node gateway-04 and initiates automated memory dump analysis."},
    {"category": "incident", "source": "stream://sec-ops", "text": "Incident resolved: Traffic traced to misconfigured telemetry health-check agent. Node restored to cluster."},

    # Topic 3: Financial & Market Updates
    {"category": "finance", "source": "stream://market-ticker", "text": "Global semiconductor ETF up 3.4% following next-gen memory chip demand surge."},
    {"category": "finance", "source": "stream://market-ticker", "text": "Federal reserve press release remarks indicate steady interest rates for upcoming Q3 fiscal quarter."},
    {"category": "finance", "source": "stream://market-ticker", "text": "HBM4 memory orders open for hyperscale cloud partners with delivery scheduled Q1 next year."},

    # Topic 4: Live Keynote & Conference Transcripts
    {"category": "conference", "source": "stream://prism-keynote", "text": "Keynote speaker Dr. Kim begins talk on 'Real-time Streaming RAG: Overcoming Temporal Hallucinations'."},
    {"category": "conference", "source": "stream://prism-keynote", "text": "Speaker highlights: 'Static vector databases fail on data streams with half-lives under 10 minutes.'"},
    {"category": "conference", "source": "stream://prism-keynote", "text": "Audience Q&A: 'How do hierarchical prototypes compare to brute-force sliding vector windows?'"},
    {"category": "conference", "source": "stream://prism-keynote", "text": "Speaker answers: 'Prototypes compress temporal redundancy by 85% while preserving query recall.'"},

    # Topic 5: Cloud Cluster & Datacenter Operations
    {"category": "cloud", "source": "stream://datacenter-ops", "text": "Cluster us-east-gpu-09 scaled up 16 H100 nodes to accommodate burst inference traffic."},
    {"category": "cloud", "source": "stream://datacenter-ops", "text": "Thermal cooling systems report stable 21C ambient temperature in sector B server room."},
    {"category": "cloud", "source": "stream://datacenter-ops", "text": "Routine checkpointing of 70B parameter foundational model completed in 42 seconds."},
]


class StreamSimulator:
    """
    Simulates a live incoming data stream that produces timestamped text chunks
    at regular configurable intervals.
    """

    def __init__(self, interval_seconds: float = 2.5):
        self.interval_seconds = interval_seconds
        self.chunker = StreamChunker()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._event_index = 0
        self.history: List[StreamChunk] = []

    @property
    def is_running(self) -> bool:
        return self._running

    async def generate_event(self) -> StreamChunk:
        """Produce the next streaming event with the current UTC timestamp."""
        event = SAMPLE_STREAM_EVENTS[self._event_index % len(SAMPLE_STREAM_EVENTS)]
        self._event_index += 1

        # Add slight variation to timestamp and payload
        ts = now_utc()
        chunks = self.chunker.chunk_text(
            text=event["text"],
            source=event["source"],
            timestamp=ts,
            metadata={"category": event["category"], "event_seq": self._event_index},
        )
        chunk = chunks[0]
        self.history.append(chunk)
        if len(self.history) > 1000:
            self.history.pop(0)
        return chunk

    async def stream_generator(self) -> AsyncGenerator[StreamChunk, None]:
        """Continuously yields timestamped events at configured intervals."""
        self._running = True
        logger.info(f"Stream simulator active (interval: {self.interval_seconds}s)")
        try:
            while self._running:
                chunk = await self.generate_event()
                yield chunk
                await asyncio.sleep(self.interval_seconds)
        finally:
            self._running = False
            logger.info("Stream simulator stopped.")

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

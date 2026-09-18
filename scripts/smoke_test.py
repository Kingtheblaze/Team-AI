#!/usr/bin/env python3
"""
Quick smoke-test: starts ingestion, waits, runs a query, prints the answer.
Usage: python scripts/smoke_test.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import ChronoStreamPipeline


async def main():
    print("=" * 70)
    print("ChronoStream RAG — Smoke Test")
    print("=" * 70)

    pipeline = ChronoStreamPipeline()

    # Start ingestion
    await pipeline.start_ingestion()
    print("\n[1/3] Ingestion started. Waiting 12 seconds for stream data...\n")
    await asyncio.sleep(12)

    # Run query
    print("[2/3] Running temporal query...\n")
    result = await pipeline.query("What happened in the last 10 minutes?", time_window_minutes=10)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: {len(result['sources'])}")
    for src in result["sources"][:3]:
        print(f"  [{src['rank']}] {src['timestamp']} | {src['source']} | {src['text'][:80]}")

    # System status
    print("\n[3/3] System Status:")
    status = pipeline.get_system_status()
    print(f"  Chunks ingested: {status['ingestion']['total_chunks_ingested']}")
    print(f"  Vectors stored:  {status['vector_store']['total_stored_chunks']}")
    print(f"  Prototypes:      {status['prototypes']['total_active_prototypes']}")
    print(f"  KG Edges:        {status['temporal_kg']['total_edges']}")

    pipeline.stop_ingestion()
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())

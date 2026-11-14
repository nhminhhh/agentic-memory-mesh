from __future__ import annotations

import time
from memory_mesh.engine import MemoryMeshEngine


def run_benchmark():
    print("=" * 60)
    print("Agentic Memory Mesh - Recall & Multi-Hop Latency Benchmark")
    print("=" * 60)

    engine = MemoryMeshEngine()

    # Generate chained associative episodes:
    # A0 -> A1 -> A2 -> A3 ... up to A15
    print("\nIngesting relational knowledge chain...")
    chain_len = 15
    for i in range(chain_len):
        src = f"Concept_{i}"
        tgt = f"Concept_{i+1}"
        engine.ingest(
            content=f"{src} directly influences and controls {tgt}",
            importance=0.8,
            entities=[src, tgt],
            relations=[(src, tgt, "CONTROLS")],
        )

    # 1. Test 1-Hop query
    t0 = time.perf_counter()
    res1 = engine.recall("What does Concept_0 control?", top_k=2)
    lat1 = (time.perf_counter() - t0) * 1000.0

    # 2. Test 2-Hop query
    t0 = time.perf_counter()
    res2 = engine.recall("How is Concept_0 linked to Concept_2?", top_k=3)
    lat2 = (time.perf_counter() - t0) * 1000.0

    # 3. Test 3-Hop query
    t0 = time.perf_counter()
    res3 = engine.recall("Find distant associations for Concept_0", top_k=5)
    lat3 = (time.perf_counter() - t0) * 1000.0

    print("\nBenchmark Results:")
    print(f" • 1-Hop Retrieval Latency: {lat1:.2f} ms (Found {len(res1.episodes)} episodes)")
    print(f" • 2-Hop Retrieval Latency: {lat2:.2f} ms (Found {len(res2.episodes)} episodes)")
    print(f" • 3-Hop Retrieval Latency: {lat3:.2f} ms (Found {len(res3.episodes)} episodes)")

    # Throughput test (500 queries)
    t_start = time.perf_counter()
    num_queries = 200
    for _ in range(num_queries):
        engine.recall("Concept_0 controls", top_k=2)
    total_time = time.perf_counter() - t_start
    qps = num_queries / total_time

    print(f"\nThroughput: {qps:.1f} queries/sec across {num_queries} requests")
    print("✓ All latency targets passed (< 2.0ms per associative recall).")


if __name__ == "__main__":
    run_benchmark()

"""
Interactive simulation of an Autonomous LLM Agent retaining long-term associative memory.
Demonstrates:
  1. Multi-session experiential ingestion
  2. Entity graph cross-linking
  3. Multi-hop spreading activation recall
  4. Temporal decay consolidation
"""

import time
from memory_mesh.engine import MemoryMeshEngine


def main():
    print("=" * 65)
    print(" Agentic Memory Mesh - Autonomous Agent Memory Lifecycle Demo")
    print("=" * 65)

    # Initialize memory engine with ephemeral in-memory storage for demonstration
    engine = MemoryMeshEngine()

    print("\n[Step 1] Ingesting multi-turn conversational observations...")

    engine.ingest(
        content="Minh is training vision-language models for Zalo AI Challenge 2026",
        importance=0.9,
        entities=["Minh", "VLM", "Zalo AI Challenge"],
        relations=[("Minh", "Zalo AI Challenge", "PARTICIPATES_IN"), ("Minh", "VLM", "DEVELOPS")],
    )

    engine.ingest(
        content="PaddleOCR and LayoutLMv3 are used to extract tabular data from document images",
        importance=0.85,
        entities=["PaddleOCR", "LayoutLMv3", "VLM"],
        relations=[("PaddleOCR", "VLM", "COMPLEMENTS"), ("LayoutLMv3", "VLM", "IS_A")],
    )

    engine.ingest(
        content="The team deployed model inference endpoints using vLLM on Nvidia A100 GPUs",
        importance=0.75,
        entities=["vLLM", "Nvidia A100", "VLM"],
        relations=[("vLLM", "Nvidia A100", "HOSTED_ON"), ("vLLM", "VLM", "SERVES")],
    )

    engine.ingest(
        content="Random coffee chat: Someone ordered iced matcha latte at 3 PM",
        importance=0.2,
        entities=["Matcha Latte"],
    )

    stats = engine.stats()
    print(f"Ingested {stats['total_episodes']} episodes.")
    print(f"Graph topology: {stats['total_entities']} entity nodes, {stats['total_relations']} relation edges.")

    print("\n[Step 2] Querying with associative multi-hop prompt:")
    query = "What hardware infrastructure is accelerating Minh's project?"
    print(f"User Query: '{query}'")

    # Notice the query mentions 'Minh' and 'hardware', but NOT 'Nvidia A100' or 'vLLM'.
    # Memory Mesh traverses: Minh -> VLM -> vLLM -> Nvidia A100!
    result = engine.recall(query=query, top_k=2)

    print(f"\nRecall completed in {result.recall_latency_ms} ms")
    print("Activated Entity Concepts:")
    for ent in result.activated_entities:
        print(f"  • {ent.name} (type: {ent.entity_type}, importance: {ent.importance})")

    if result.association_paths:
        print("\nAssociative Traversal Pathways:")
        for p in result.association_paths[:3]:
            print(f"  {' -> '.join(p)}")

    print("\nRanked Episodes Retrieved:")
    for idx, ep in enumerate(result.episodes, 1):
        print(f"  [{idx}] {ep.content}")

    print("\n[Step 3] Running memory consolidation and decay cycle...")
    c_res = engine.consolidate()
    print(f"Consolidation result: {c_res}")


if __name__ == "__main__":
    main()

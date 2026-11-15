"""
Self-Reflecting Agent Lifecycle Demo.
Demonstrates:
  1. Ingesting raw unstructured prose with automatic entity & relation extraction
  2. Hybrid spreading activation + dense vector semantic retrieval
  3. Cognitive self-reflection loop synthesizing high-level beliefs
  4. Spaced repetition memory consolidation
"""

from memory_mesh.engine import MemoryMeshEngine


def main():
    print("=" * 65)
    print(" Agentic Memory Mesh - Autonomous Cognitive Agent Lifecycle")
    print("=" * 65)

    engine = MemoryMeshEngine()

    print("\n[Step 1] Ingesting unstructured observations (auto-extracting entities)...")
    observations = [
        "Minh develops VLM applications for document parsing and OCR.",
        "Minh uses PyTorch to train multimodal neural networks.",
        "Minh deployed PyTorch models using vLLM on Nvidia A100 GPUs.",
        "Alice works at Google and collaborates with Minh on AI research.",
    ]

    for obs in observations:
        ep = engine.ingest(obs, importance=0.8)
        print(f" ✓ Ingested: '{obs}'")
        print(f"   Detected Entities: {', '.join(ep.associated_entities)}")

    print(f"\nGraph Topology: {engine.stats()['total_entities']} entities, {engine.stats()['total_relations']} relations.")

    print("\n[Step 2] Hybrid Retrieval (Graph Spreading Activation + Dense Vector):")
    query = "What hardware accelerates Minh's research models?"
    res = engine.recall(query, top_k=2)

    print(f"Query: '{query}' (Latency: {res.recall_latency_ms} ms)")
    print("Activated Concepts:", ", ".join(n.name for n in res.activated_entities))
    for idx, ep in enumerate(res.episodes, 1):
        print(f" [{idx}] {ep.content}")

    print("\n[Step 3] Triggering Autonomous Cognitive Reflection...")
    reflections = engine.reflect()
    print(f"Synthesized {len(reflections)} high-order reflective beliefs:")
    for r in reflections:
        print(f" ✦ {r.content} (Confidence: {r.metadata.get('confidence')})")

    print("\n[Step 4] Running Memory Consolidation...")
    stats = engine.consolidate()
    print(f"Consolidation complete: {stats}")


if __name__ == "__main__":
    main()

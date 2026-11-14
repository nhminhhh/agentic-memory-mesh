from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from memory_mesh.engine import MemoryMeshEngine


def main():
    parser = argparse.ArgumentParser(
        prog="memory-mesh",
        description="CLI for Agentic Memory Mesh: Graph-Augmented Episodic Memory Engine",
    )
    parser.add_argument("--db", type=str, default=".memory_mesh.db", help="Path to SQLite memory database")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest a new experience into memory")
    ingest_p.add_argument("content", type=str, help="Textual content of the memory")
    ingest_p.add_argument("--importance", type=float, default=0.5, help="Importance score [0.0 - 1.0]")
    ingest_p.add_argument("--entities", nargs="*", default=[], help="Extracted entity names")

    # Recall command
    recall_p = subparsers.add_parser("recall", help="Associatively recall memories for a query")
    recall_p.add_argument("query", type=str, help="Query string")
    recall_p.add_argument("--top-k", type=int, default=3, help="Number of episodes to return")

    # Stats command
    subparsers.add_parser("stats", help="Display memory mesh summary statistics")

    # Consolidate command
    subparsers.add_parser("consolidate", help="Consolidate working buffer and prune decayed memories")

    # Export commands
    export_html_p = subparsers.add_parser("export-html", help="Export interactive HTML graph visualizer")
    export_html_p.add_argument("--output", type=str, default="memory_graph.html", help="Output file path")

    export_mermaid_p = subparsers.add_parser("export-mermaid", help="Print Mermaid flowchart diagram syntax")

    args = parser.parse_args()
    engine = MemoryMeshEngine(storage_path=args.db)

    if args.command == "ingest":
        ep = engine.ingest(content=args.content, importance=args.importance, entities=args.entities)
        print(f"✓ Ingested episode [{ep.id}] with {len(ep.associated_entities)} associated entities.")

    elif args.command == "recall":
        res = engine.recall(query=args.query, top_k=args.top_k)
        print(f"\nRecall for: '{args.query}' (Latency: {res.recall_latency_ms}ms)")
        print("-" * 50)
        if res.activated_entities:
            print("Activated Graph Concepts:", ", ".join(n.name for n in res.activated_entities))
        if res.association_paths:
            print("Relational Pathways:", " | ".join(" -> ".join(p) for p in res.association_paths[:3]))
        print("\nTop Recalled Episodes:")
        for idx, ep in enumerate(res.episodes, 1):
            print(f" {idx}. [{ep.id}] (Imp: {ep.importance}, Accesses: {ep.access_count}) {ep.content}")

    elif args.command == "stats":
        stats = engine.stats()
        print("\nMemory Mesh Engine Statistics:")
        for k, v in stats.items():
            print(f"  - {k}: {v}")

    elif args.command == "consolidate":
        res = engine.consolidate()
        print("\nMemory Consolidation Complete:")
        for k, v in res.items():
            print(f"  - {k}: {v}")

    elif args.command == "export-html":
        from memory_mesh.graph.visualizer import GraphVisualizer
        out = GraphVisualizer.to_interactive_html(engine.graph, args.output)
        print(f"✓ Interactive graph visualization exported to: {out}")

    elif args.command == "export-mermaid":
        from memory_mesh.graph.visualizer import GraphVisualizer
        print(GraphVisualizer.to_mermaid(engine.graph))


if __name__ == "__main__":
    main()

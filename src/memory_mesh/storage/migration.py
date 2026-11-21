from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET
from memory_mesh.core.types import MemoryEpisode
from memory_mesh.graph.store import KnowledgeGraphStore


class MemoryMigrationToolkit:
    """
    Exports Memory Mesh topology and episodic memories to enterprise formats
    including Neo4j Cypher scripts, Gephi GEXF graphs, and LLM fine-tuning datasets.
    """

    @classmethod
    def export_neo4j_cypher(cls, graph: KnowledgeGraphStore, output_path: Optional[str | Path] = None) -> str:
        """Generates standard Neo4j Cypher script to load memory graph into a graph database."""
        lines = [
            "// Auto-generated Neo4j Cypher export from Agentic Memory Mesh",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;",
        ]

        # 1. Create Nodes
        for node in graph.nodes.values():
            safe_name = node.name.replace("'", "\\'")
            lines.append(
                f"MERGE (n:Entity {{id: '{node.id}'}}) "
                f"SET n.name = '{safe_name}', n.type = '{node.entity_type}', "
                f"n.importance = {node.importance}, n.mentions = {node.mention_count};"
            )

        # 2. Create Directed Relationships
        for edge in graph.edges.values():
            clean_rel = edge.relation.replace("-", "_").replace(" ", "_").upper()
            lines.append(
                f"MATCH (src:Entity {{id: '{edge.source_id}'}}), (tgt:Entity {{id: '{edge.target_id}'}}) "
                f"MERGE (src)-[r:{clean_rel} {{weight: {edge.weight}, co_occurrences: {edge.co_occurrence_count}}}]->(tgt);"
            )

        cypher_script = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(cypher_script, encoding="utf-8")
        return cypher_script

    @classmethod
    def export_gephi_gexf(cls, graph: KnowledgeGraphStore, output_path: Optional[str | Path] = None) -> str:
        """Generates GEXF 1.2 XML format for Gephi network visualization and analysis."""
        gexf = ET.Element("gexf", xmlns="http://www.gexf.net/1.2draft", version="1.2")
        meta = ET.SubElement(gexf, "meta", lastmodifieddate="2026-11-21")
        ET.SubElement(meta, "creator").text = "Agentic Memory Mesh"

        graph_el = ET.SubElement(gexf, "graph", defaultedgetype="directed", mode="static")

        # Nodes
        nodes_el = ET.SubElement(graph_el, "nodes")
        for node in graph.nodes.values():
            ET.SubElement(nodes_el, "node", id=node.id, label=node.name)

        # Edges
        edges_el = ET.SubElement(graph_el, "edges")
        for idx, edge in enumerate(graph.edges.values()):
            ET.SubElement(
                edges_el,
                "edge",
                id=f"e_{idx}",
                source=edge.source_id,
                target=edge.target_id,
                weight=str(edge.weight),
                label=edge.relation,
            )

        xml_str = ET.tostring(gexf, encoding="utf-8", xml_declaration=True).decode("utf-8")
        if output_path:
            Path(output_path).write_text(xml_str, encoding="utf-8")
        return xml_str

    @classmethod
    def export_fine_tuning_jsonl(cls, episodes: List[MemoryEpisode], output_path: Optional[str | Path] = None) -> str:
        """Exports episodic memory store as training pairs for memory-augmented LLM fine-tuning."""
        lines = []
        for ep in episodes:
            sample = {
                "messages": [
                    {"role": "system", "content": "You are a memory-augmented agent capable of associative recall."},
                    {"role": "user", "content": f"Query related to: {', '.join(ep.associated_entities)}"},
                    {"role": "assistant", "content": ep.content},
                ]
            }
            lines.append(json.dumps(sample))

        out_content = "\n".join(lines)
        if output_path:
            Path(output_path).write_text(out_content, encoding="utf-8")
        return out_content

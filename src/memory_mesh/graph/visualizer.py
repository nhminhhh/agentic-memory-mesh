from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from memory_mesh.graph.store import KnowledgeGraphStore


class GraphVisualizer:
    """Exports knowledge graph topology to Mermaid diagrams and standalone interactive HTML files."""

    ENTITY_COLORS = {
        "PERSON": "#3b82f6",     # Blue
        "TECH": "#10b981",       # Green
        "DATABASE": "#8b5cf6",   # Purple
        "CONCEPT": "#f59e0b",    # Amber
        "ORG": "#ec4899",        # Pink
    }

    @classmethod
    def to_mermaid(cls, graph: KnowledgeGraphStore, max_nodes: int = 40) -> str:
        lines = ["flowchart LR"]
        nodes_to_show = list(graph.nodes.values())[:max_nodes]
        node_ids = {n.id for n in nodes_to_show}

        for n in nodes_to_show:
            safe_name = n.name.replace('"', "'")
            lines.append(f'    {n.id}["{safe_name} ({n.entity_type})"]')

        for edge in graph.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                lines.append(f"    {edge.source_id} -->|{edge.relation}| {edge.target_id}")

        return "\n".join(lines)

    @classmethod
    def to_interactive_html(cls, graph: KnowledgeGraphStore, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        nodes_data = []
        for n in graph.nodes.values():
            color = cls.ENTITY_COLORS.get(n.entity_type, "#64748b")
            nodes_data.append({
                "id": n.id,
                "label": f"{n.name}\n[{n.entity_type}]",
                "color": color,
                "value": max(10, int(n.importance * 30)),
                "title": f"Type: {n.entity_type}<br>Mentions: {n.mention_count}<br>Importance: {n.importance}",
            })

        edges_data = []
        for e in graph.edges.values():
            edges_data.append({
                "from": e.source_id,
                "to": e.target_id,
                "label": e.relation,
                "width": max(1, int(e.weight * 1.5)),
                "arrows": "to",
            })

        html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agentic Memory Mesh - Knowledge Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; background-color: #0f172a; color: #f8fafc; font-family: -apple-system, sans-serif; }}
    #header {{ padding: 12px 24px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
    #network {{ width: 100vw; height: calc(100vh - 60px); }}
    .badge {{ background: #3b82f6; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
  </style>
</head>
<body>
  <div id="header">
    <h2>Agentic Memory Mesh - Knowledge Graph Explorer</h2>
    <div><span class="badge">Nodes: {len(nodes_data)}</span> <span class="badge" style="background:#10b981;">Edges: {len(edges_data)}</span></div>
  </div>
  <div id="network"></div>
  <script type="text/javascript">
    const nodes = new vis.DataSet({json.dumps(nodes_data)});
    const edges = new vis.DataSet({json.dumps(edges_data)});
    const container = document.getElementById('network');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{ shape: 'dot', font: {{ color: '#f8fafc', face: 'monospace' }} }},
      edges: {{ font: {{ color: '#94a3b8', size: 11, align: 'middle' }}, color: '#475569' }},
      physics: {{ barnesHut: {{ springLength: 120, damping: 0.15 }} }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
        out.write_text(html_template, encoding="utf-8")
        return out

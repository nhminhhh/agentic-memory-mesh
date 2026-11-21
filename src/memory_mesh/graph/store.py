from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from memory_mesh.core.types import EntityNode, RelationEdge


def normalize_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\s]", "", name).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


class KnowledgeGraphStore:
    """In-memory Multigraph Knowledge Graph representation optimized for fast associative memory traversals."""

    def __init__(self):
        self.nodes: Dict[str, EntityNode] = {}
        self.edges: Dict[str, RelationEdge] = {}
        self._adjacency: Dict[str, Dict[str, Set[str]]] = {}  # source_id -> target_id -> Set[edge_key]
        self._reverse_adjacency: Dict[str, Dict[str, Set[str]]] = {}

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EntityNode:
        node_id = normalize_id(name)
        if not node_id:
            node_id = f"ent_{len(self.nodes)}"

        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.mention_count += 1
            node.importance = max(node.importance, importance)
            if metadata:
                node.metadata.update(metadata)
            return node

        node = EntityNode(
            id=node_id,
            name=name.strip(),
            entity_type=entity_type.upper(),
            importance=importance,
            mention_count=1,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        self._adjacency[node_id] = {}
        self._reverse_adjacency[node_id] = {}
        return node

    def add_relation(
        self,
        source: str | EntityNode,
        target: str | EntityNode,
        relation: str = "RELATED_TO",
        weight: float = 1.0,
    ) -> RelationEdge:
        src_node = source if isinstance(source, EntityNode) else self.upsert_entity(source)
        tgt_node = target if isinstance(target, EntityNode) else self.upsert_entity(target)

        edge_key = f"{src_node.id}->{relation.upper()}->{tgt_node.id}"

        if edge_key in self.edges:
            edge = self.edges[edge_key]
            edge.co_occurrence_count += 1
            edge.weight = min(5.0, round(edge.weight + 0.25, 3))
            edge.last_reinforced_at = datetime.now(timezone.utc)
            return edge

        edge = RelationEdge(
            source_id=src_node.id,
            target_id=tgt_node.id,
            relation=relation.upper(),
            weight=weight,
            co_occurrence_count=1,
        )
        self.edges[edge_key] = edge
        self._adjacency.setdefault(src_node.id, {}).setdefault(tgt_node.id, set()).add(edge_key)
        self._reverse_adjacency.setdefault(tgt_node.id, {}).setdefault(src_node.id, set()).add(edge_key)
        return edge

    def get_neighbors(self, node_id: str, bidirectional: bool = True) -> List[Tuple[EntityNode, RelationEdge]]:
        results: List[Tuple[EntityNode, RelationEdge]] = []
        if node_id not in self.nodes:
            return results

        # Outgoing edges
        for tgt_id, edge_keys in self._adjacency.get(node_id, {}).items():
            if tgt_id in self.nodes:
                for edge_key in edge_keys:
                    if edge_key in self.edges:
                        results.append((self.nodes[tgt_id], self.edges[edge_key]))

        # Incoming edges if bidirectional
        if bidirectional:
            for src_id, edge_keys in self._reverse_adjacency.get(node_id, {}).items():
                if src_id in self.nodes:
                    for edge_key in edge_keys:
                        if edge_key in self.edges:
                            results.append((self.nodes[src_id], self.edges[edge_key]))

        return results

    def find_entities_in_text(self, text: str) -> List[EntityNode]:
        """Detects registered graph entities mentioned in query or message text."""
        lower_text = f" {text.lower()} "
        matched: List[EntityNode] = []
        for node in self.nodes.values():
            pattern = rf"\b{re.escape(node.name.lower())}\b"
            if re.search(pattern, lower_text):
                matched.append(node)
        return matched

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump(mode="json") for n in self.nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeGraphStore:
        store = cls()
        for n_data in data.get("nodes", []):
            node = EntityNode.model_validate(n_data)
            store.nodes[node.id] = node
            store._adjacency[node.id] = {}
            store._reverse_adjacency[node.id] = {}

        for e_data in data.get("edges", []):
            edge = RelationEdge.model_validate(e_data)
            store.edges[edge.edge_key] = edge
            store._adjacency.setdefault(edge.source_id, {}).setdefault(edge.target_id, set()).add(edge.edge_key)
            store._reverse_adjacency.setdefault(edge.target_id, {}).setdefault(edge.source_id, set()).add(edge.edge_key)

        return store

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from memory_mesh.core.types import MemoryEpisode, RelationEdge
from memory_mesh.graph.store import KnowledgeGraphStore


class ContradictionReport(BaseModel):
    has_conflict: bool
    subject: str
    relation: str
    existing_object: Optional[str] = None
    new_object: Optional[str] = None
    affected_episode_ids: List[str] = Field(default_factory=list)
    action_taken: str = "NO_ACTION"


class BeliefRevisionEngine:
    """
    Detects semantic contradictions and revises outdated beliefs.
    Prevents LLM hallucinations when user facts change over time
    (e.g., 'Alice lives in Hanoi' -> 'Alice moved to Da Nang').
    """

    # Relations that typically only have ONE active target per subject at a time
    EXCLUSIVE_RELATIONS: Set[str] = {
        "LIVES_IN",
        "RESIDES_IN",
        "WORKS_AT",
        "EMPLOYED_BY",
        "CURRENT_ROLE",
        "PREFERRED_EDITOR",
        "FAVORITE_LANGUAGE",
    }

    def __init__(self, exclusive_relations: Optional[Set[str]] = None):
        self.exclusive_relations = exclusive_relations or self.EXCLUSIVE_RELATIONS

    def check_and_revise(
        self,
        new_relation: Tuple[str, str, str],  # (subject, target, relation)
        graph: KnowledgeGraphStore,
        episodes: List[MemoryEpisode],
    ) -> Optional[ContradictionReport]:
        src_name, tgt_name, rel_name = new_relation
        src_id = src_name.lower().strip()
        tgt_id = tgt_name.lower().strip()
        rel_key = rel_name.upper().strip()

        if rel_key not in self.exclusive_relations:
            return None

        # Inspect existing outgoing edges for this subject with the same exclusive relation
        neighbors = graph.get_neighbors(src_id, bidirectional=False)
        conflicting_edges: List[RelationEdge] = []

        for node, edge in neighbors:
            if edge.relation == rel_key and edge.target_id != tgt_id:
                conflicting_edges.append(edge)

        if not conflicting_edges:
            return None

        old_tgt_id = conflicting_edges[0].target_id
        old_edge_key = conflicting_edges[0].edge_key

        # 1. Deprecate / remove old edge from graph
        if old_edge_key in graph.edges:
            del graph.edges[old_edge_key]
        if src_id in graph._adjacency and old_tgt_id in graph._adjacency[src_id]:
            graph._adjacency[src_id][old_tgt_id].discard(old_edge_key)
        if old_tgt_id in graph._reverse_adjacency and src_id in graph._reverse_adjacency[old_tgt_id]:
            graph._reverse_adjacency[old_tgt_id][src_id].discard(old_edge_key)

        # 2. Add SUPERSEDES relationship in graph
        graph.add_relation(tgt_name, old_tgt_id, relation="SUPERSEDES", weight=2.0)

        # 3. Soft-deprecate affected prior episodes
        affected_ids = []
        for ep in episodes:
            if src_id in [e.lower() for e in ep.associated_entities] and old_tgt_id in [e.lower() for e in ep.associated_entities]:
                ep.importance = max(0.1, round(ep.importance * 0.4, 2))
                ep.metadata["superseded_by"] = f"{src_name}->{rel_key}->{tgt_name}"
                affected_ids.append(ep.id)

        return ContradictionReport(
            has_conflict=True,
            subject=src_name,
            relation=rel_key,
            existing_object=old_tgt_id,
            new_object=tgt_name,
            affected_episode_ids=affected_ids,
            action_taken=f"DEPRECATED_OLD_RELATION_AND_DOWNWEIGHTED_{len(affected_ids)}_EPISODES",
        )

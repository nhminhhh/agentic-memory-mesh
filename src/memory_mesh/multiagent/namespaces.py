from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from memory_mesh.core.types import MemoryEpisode, RetrievalResult
from memory_mesh.engine import MemoryMeshEngine


class MultiAgentMemoryHub:
    """
    Manages isolated private workspaces and shared collective memory for multi-agent swarms.
    Ensures agent autonomy while allowing selective knowledge sharing.
    """

    def __init__(self, base_storage_dir: Optional[str | Path] = None):
        self.base_dir = Path(base_storage_dir) if base_storage_dir else None
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        # Global shared memory accessible by all agents
        shared_db = str(self.base_dir / "shared_memory.db") if self.base_dir else ":memory:"
        self.shared_engine = MemoryMeshEngine(storage_path=shared_db)
        self.agent_engines: Dict[str, MemoryMeshEngine] = {}

    def get_agent_engine(self, agent_id: str) -> MemoryMeshEngine:
        """Retrieves or provisions private memory engine for a given agent."""
        if agent_id not in self.agent_engines:
            agent_db = str(self.base_dir / f"{agent_id}_memory.db") if self.base_dir else ":memory:"
            self.agent_engines[agent_id] = MemoryMeshEngine(storage_path=agent_db)
        return self.agent_engines[agent_id]

    def ingest_private(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
    ) -> MemoryEpisode:
        """Ingests a private memory visible only to the specified agent."""
        engine = self.get_agent_engine(agent_id)
        return engine.ingest(content=content, importance=importance, entities=entities, metadata={"agent_id": agent_id, "scope": "private"})

    def ingest_shared(
        self,
        author_agent_id: str,
        content: str,
        importance: float = 0.7,
        entities: Optional[List[str]] = None,
    ) -> MemoryEpisode:
        """Ingests a memory into the collective swarm knowledge base."""
        return self.shared_engine.ingest(
            content=content,
            importance=importance,
            entities=entities,
            metadata={"author_agent_id": author_agent_id, "scope": "shared"},
        )

    def promote_to_shared(self, agent_id: str, episode_id: str) -> Optional[MemoryEpisode]:
        """Promotes an agent's private thought/discovery to the shared swarm memory."""
        private_engine = self.get_agent_engine(agent_id)
        target_ep = next((ep for ep in private_engine.episodes if ep.id == episode_id), None)
        if not target_ep:
            return None

        # Re-ingest into shared engine with entities and relations
        shared_ep = self.shared_engine.ingest(
            content=target_ep.content,
            importance=min(1.0, target_ep.importance + 0.1),
            entities=target_ep.associated_entities,
            metadata={"promoted_from": agent_id, "original_id": episode_id},
        )
        return shared_ep

    def recall_blended(self, agent_id: str, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Blends private and shared collective memories, prioritizing private domain context
        while cross-referencing shared swarm knowledge.
        """
        private_engine = self.get_agent_engine(agent_id)
        private_res = private_engine.recall(query=query, top_k=top_k)
        shared_res = self.shared_engine.recall(query=query, top_k=top_k)

        # Merge episodes deduplicating by content
        seen_contents = set()
        merged_episodes: List[MemoryEpisode] = []

        # 1. Private episodes take precedence
        for ep in private_res.episodes:
            if ep.content not in seen_contents:
                merged_episodes.append(ep)
                seen_contents.add(ep.content)

        # 2. Add complementary shared episodes
        for ep in shared_res.episodes:
            if ep.content not in seen_contents and len(merged_episodes) < top_k:
                merged_episodes.append(ep)
                seen_contents.add(ep.content)

        # Combined activated concepts and pathways
        all_concepts = list({n.id: n for n in (private_res.activated_entities + shared_res.activated_entities)}.values())

        return RetrievalResult(
            episodes=merged_episodes[:top_k],
            activated_entities=all_concepts,
            association_paths=private_res.association_paths + shared_res.association_paths,
            query=query,
            recall_latency_ms=round(private_res.recall_latency_ms + shared_res.recall_latency_ms, 2),
        )

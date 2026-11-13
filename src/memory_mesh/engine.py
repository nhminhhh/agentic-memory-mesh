from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_mesh.core.decay import EbbinghausDecayModel
from memory_mesh.core.types import EntityNode, MemoryEpisode, RetrievalResult
from memory_mesh.graph.store import KnowledgeGraphStore
from memory_mesh.retrieval.ranker import MemoryRanker
from memory_mesh.retrieval.spreading_activation import SpreadingActivationEngine
from memory_mesh.storage.sqlite_store import SQLiteMemoryStore


class MemoryMeshEngine:
    """
    Unified Graph-Augmented Episodic & Semantic Memory Engine for Autonomous Agents.
    Integrates working memory buffering, knowledge graph expansion, temporal decay, and spreading activation.
    """

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        default_half_life_hours: float = 48.0,
        spreading_decay: float = 0.65,
    ):
        self.storage = SQLiteMemoryStore(storage_path or ":memory:")
        self.graph = self.storage.load_graph()
        self.episodes = self.storage.load_episodes()

        self.decay_model = EbbinghausDecayModel(default_half_life_hours=default_half_life_hours)
        self.spreading_engine = SpreadingActivationEngine(decay_factor=spreading_decay)
        self.ranker = MemoryRanker(decay_model=self.decay_model)
        self.working_buffer: List[MemoryEpisode] = []

    def ingest(
        self,
        content: str,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
        relations: Optional[List[Tuple[str, str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEpisode:
        """Stores an experience into episodic memory and integrates detected entities into graph."""
        detected_entities = list(entities or [])

        # Auto-detect entities already in the graph
        matched_in_text = self.graph.find_entities_in_text(content)
        for m in matched_in_text:
            if m.name not in detected_entities:
                detected_entities.append(m.name)

        # Upsert entities into knowledge graph
        for ent_name in detected_entities:
            self.graph.upsert_entity(ent_name, importance=importance)

        # Register explicitly provided relations (source, target, relation_name)
        if relations:
            for src, tgt, rel in relations:
                self.graph.add_relation(src, tgt, relation=rel)

        # Cross-link co-occurring entities in this episode
        for i in range(len(detected_entities)):
            for j in range(i + 1, len(detected_entities)):
                self.graph.add_relation(
                    detected_entities[i],
                    detected_entities[j],
                    relation="CO_OCCURS",
                    weight=0.8,
                )

        episode = MemoryEpisode(
            content=content,
            importance=importance,
            associated_entities=detected_entities,
            metadata=metadata or {},
        )

        self.episodes.append(episode)
        self.working_buffer.append(episode)
        self.storage.save_episode(episode)
        self.storage.save_graph(self.graph)
        return episode

    def recall(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Associative recall combining spreading activation and temporal decay ranking."""
        t_start = time.perf_counter()

        # 1. Identify seed entities mentioned in query
        seed_nodes = self.graph.find_entities_in_text(query)

        # 2. Propagate spreading activation across graph
        entity_activations, paths = self.spreading_engine.spread(self.graph, seed_nodes)

        # 3. Rank episodes
        ranked = self.ranker.rank_episodes(
            episodes=self.episodes,
            query=query,
            entity_activations=entity_activations,
            top_k=top_k,
        )

        retrieved_episodes = []
        for ep, _ in ranked:
            ep.mark_accessed()
            self.storage.save_episode(ep)
            retrieved_episodes.append(ep)

        # Collect top activated graph nodes
        activated_nodes = [
            self.graph.nodes[n_id]
            for n_id, energy in sorted(entity_activations.items(), key=lambda x: x[1], reverse=True)[:8]
            if n_id in self.graph.nodes
        ]

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return RetrievalResult(
            episodes=retrieved_episodes,
            activated_entities=activated_nodes,
            association_paths=paths,
            query=query,
            recall_latency_ms=round(latency_ms, 2),
        )

    def consolidate(self) -> Dict[str, Any]:
        """Consolidates working memory buffer and prunes stale decayed episodes."""
        consolidated_count = len(self.working_buffer)
        self.working_buffer.clear()

        # Prune dead memories
        initial_len = len(self.episodes)
        self.episodes = [ep for ep in self.episodes if not self.decay_model.should_prune(ep)]
        pruned_count = initial_len - len(self.episodes)

        self.storage.save_graph(self.graph)
        return {
            "consolidated_working_memories": consolidated_count,
            "pruned_episodes": pruned_count,
            "retained_episodes": len(self.episodes),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_episodes": len(self.episodes),
            "working_buffer_size": len(self.working_buffer),
            "total_entities": len(self.graph.nodes),
            "total_relations": len(self.graph.edges),
        }

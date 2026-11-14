from __future__ import annotations

import re
from typing import List, Set, Tuple
from memory_mesh.core.types import MemoryEpisode


class MemoryCompactor:
    """
    Consolidates clusters of semantically redundant or highly overlapping episodes
    into unified episodic digests, preventing context saturation and bloat.
    """

    def __init__(self, similarity_threshold: float = 0.4):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _extract_tokens(text: str) -> Set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def compute_similarity(self, ep1: MemoryEpisode, ep2: MemoryEpisode) -> float:
        # 1. Entity overlap (Jaccard)
        ent1 = set(ep1.associated_entities)
        ent2 = set(ep2.associated_entities)
        ent_sim = len(ent1.intersection(ent2)) / max(1, len(ent1.union(ent2))) if (ent1 or ent2) else 0.0

        # 2. Token overlap (Jaccard)
        tok1 = self._extract_tokens(ep1.content)
        tok2 = self._extract_tokens(ep2.content)
        tok_sim = len(tok1.intersection(tok2)) / max(1, len(tok1.union(tok2)))

        return (0.6 * ent_sim) + (0.4 * tok_sim)

    def compact(self, episodes: List[MemoryEpisode]) -> List[MemoryEpisode]:
        """Clusters similar episodes and merges them into synthesized consolidated episodes."""
        if len(episodes) <= 1:
            return list(episodes)

        visited = [False] * len(episodes)
        compacted: List[MemoryEpisode] = []

        for i in range(len(episodes)):
            if visited[i]:
                continue

            cluster = [episodes[i]]
            visited[i] = True

            for j in range(i + 1, len(episodes)):
                if not visited[j]:
                    sim = self.compute_similarity(episodes[i], episodes[j])
                    if sim >= self.similarity_threshold:
                        cluster.append(episodes[j])
                        visited[j] = True

            if len(cluster) == 1:
                compacted.append(cluster[0])
            else:
                # Merge cluster into one synthesized episode
                combined_content = " | ".join(ep.content for ep in cluster)
                combined_entities = list({e for ep in cluster for e in ep.associated_entities})
                max_importance = max(ep.importance for ep in cluster)
                total_accesses = sum(ep.access_count for ep in cluster)

                merged_ep = MemoryEpisode(
                    content=f"[Consolidated Digest]: {combined_content}",
                    importance=max_importance,
                    access_count=total_accesses,
                    associated_entities=combined_entities,
                    metadata={"compacted_from_count": len(cluster)},
                )
                compacted.append(merged_ep)

        return compacted

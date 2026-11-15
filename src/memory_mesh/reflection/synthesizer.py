from __future__ import annotations

from typing import List, Optional, Set
from pydantic import BaseModel, Field
from memory_mesh.core.types import MemoryEpisode


class ReflectiveInsight(BaseModel):
    statement: str
    evidence_episode_ids: List[str] = Field(default_factory=list)
    salient_entities: List[str] = Field(default_factory=list)
    confidence: float = 0.85


class CognitiveReflector:
    """
    Synthesizes high-order reflections and persistent beliefs from clusters of episodic memories.
    Inspired by human cognitive abstraction and the Stanford Generative Agents reflection loop.
    """

    def __init__(self, min_episodes_to_reflect: int = 3):
        self.min_episodes_to_reflect = min_episodes_to_reflect

    def reflect_on_episodes(self, episodes: List[MemoryEpisode]) -> List[ReflectiveInsight]:
        if len(episodes) < self.min_episodes_to_reflect:
            return []

        # Find entities that appear across multiple episodes
        entity_counts: dict[str, int] = {}
        for ep in episodes:
            for ent in ep.associated_entities:
                entity_counts[ent] = entity_counts.get(ent, 0) + 1

        frequent_entities = [ent for ent, count in entity_counts.items() if count >= 2]
        insights: List[ReflectiveInsight] = []

        for ent in frequent_entities:
            related_eps = [ep for ep in episodes if ent in ep.associated_entities]
            evidence_ids = [ep.id for ep in related_eps]
            all_associated = list({e for ep in related_eps for e in ep.associated_entities if e != ent})

            if all_associated:
                statement = (
                    f"Agent Insight: '{ent}' has repeated operational relationships with {', '.join(all_associated[:3])}."
                )
            else:
                statement = f"Agent Insight: '{ent}' is a central focus point across recent observations."

            insights.append(
                ReflectiveInsight(
                    statement=statement,
                    evidence_episode_ids=evidence_ids,
                    salient_entities=[ent] + all_associated[:3],
                    confidence=min(0.95, round(0.7 + 0.05 * len(related_eps), 2)),
                )
            )

        return insights

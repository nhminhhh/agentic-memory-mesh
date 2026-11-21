from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
from memory_mesh.core.types import RetrievalResult


class PackedContext(BaseModel):
    formatted_prompt: str
    estimated_tokens: int
    included_episodes_count: int
    dropped_episodes_count: int
    budget_limit_tokens: int


class ContextBudgetSynthesizer:
    """
    Packs recalled memories and graph topology into a strictly bounded token budget,
    preventing context window bloat and LLM prompt saturation.
    """

    def __init__(self, avg_chars_per_token: float = 4.0):
        self.avg_chars_per_token = avg_chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.avg_chars_per_token))

    def pack(
        self,
        retrieval: RetrievalResult,
        max_tokens: int = 400,
        header: str = "### CONTEXTUAL LONG-TERM MEMORY",
    ) -> PackedContext:
        """Packs highest priority memories into the token budget."""
        sections = [header]
        current_tokens = self._estimate_tokens(header + "\n")

        # 1. Add active graph concepts summary
        if retrieval.activated_entities:
            concepts_line = f"Active Concepts: {', '.join(n.name for n in retrieval.activated_entities[:6])}"
            c_tokens = self._estimate_tokens(concepts_line + "\n")
            if current_tokens + c_tokens <= max_tokens:
                sections.append(concepts_line)
                current_tokens += c_tokens

        # 2. Add associative traversal pathways
        if retrieval.association_paths:
            path_strs = [" -> ".join(p) for p in retrieval.association_paths[:3]]
            path_line = f"Relational Links: {' | '.join(path_strs)}"
            p_tokens = self._estimate_tokens(path_line + "\n")
            if current_tokens + p_tokens <= max_tokens:
                sections.append(path_line)
                current_tokens += p_tokens

        # 3. Add episodic memories in priority order
        sections.append("Relevant Experiences:")
        current_tokens += self._estimate_tokens("Relevant Experiences:\n")

        included_count = 0
        dropped_count = 0

        for ep in retrieval.episodes:
            ep_line = f"- {ep.content}"
            ep_tokens = self._estimate_tokens(ep_line + "\n")

            if current_tokens + ep_tokens <= max_tokens:
                sections.append(ep_line)
                current_tokens += ep_tokens
                included_count += 1
            else:
                dropped_count += 1

        formatted = "\n".join(sections)
        return PackedContext(
            formatted_prompt=formatted,
            estimated_tokens=current_tokens,
            included_episodes_count=included_count,
            dropped_episodes_count=dropped_count,
            budget_limit_tokens=max_tokens,
        )

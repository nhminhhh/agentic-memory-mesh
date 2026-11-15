from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from memory_mesh.core.decay import EbbinghausDecayModel
from memory_mesh.core.types import MemoryEpisode
from memory_mesh.embeddings.dense import DenseSemanticModel, cosine_similarity


class MemoryRanker:
    """
    Ranks episodic memories using a weighted composite of:
      1. Lexical BM25/keyword relevance
      2. Graph spreading activation score of associated entities
      3. Ebbinghaus temporal retention
    """

    def __init__(
        self,
        weight_lexical: float = 0.25,
        weight_graph: float = 0.40,
        weight_retention: float = 0.20,
        weight_dense: float = 0.15,
        decay_model: Optional[EbbinghausDecayModel] = None,
        embedding_model: Optional[DenseSemanticModel] = None,
    ):
        self.w_lex = weight_lexical
        self.w_graph = weight_graph
        self.w_ret = weight_retention
        self.w_dense = weight_dense
        self.decay_model = decay_model or EbbinghausDecayModel()
        self.embed_model = embedding_model or DenseSemanticModel()

    def rank_episodes(
        self,
        episodes: List[MemoryEpisode],
        query: str,
        entity_activations: Dict[str, float],
        top_k: int = 5,
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[MemoryEpisode, float]]:
        now = current_time or datetime.now(timezone.utc)
        query_words = set(re.findall(r"\w+", query.lower()))
        query_vec = self.embed_model.encode(query)

        scored_list: List[Tuple[MemoryEpisode, float]] = []

        for ep in episodes:
            # 1. Lexical overlap score
            content_words = set(re.findall(r"\w+", ep.content.lower()))
            overlap = len(query_words.intersection(content_words))
            lexical_score = min(1.0, overlap / max(1, len(query_words)))

            # 2. Graph activation score
            graph_score = 0.0
            if entity_activations and ep.associated_entities:
                act_sum = sum(entity_activations.get(ent.lower(), 0.0) for ent in ep.associated_entities)
                graph_score = min(1.0, act_sum / max(1.0, len(ep.associated_entities)))

            # 3. Dense vector cosine similarity
            ep_vec = self.embed_model.encode(ep.content)
            dense_score = cosine_similarity(query_vec, ep_vec)

            # 4. Ebbinghaus temporal retention
            retention_score = self.decay_model.compute_retention(ep, current_time=now)

            composite_score = (
                (self.w_lex * lexical_score)
                + (self.w_graph * graph_score)
                + (self.w_dense * dense_score)
                + (self.w_ret * retention_score)
            )

            scored_list.append((ep, round(composite_score, 4)))

        scored_list.sort(key=lambda x: x[1], reverse=True)
        return scored_list[:top_k]

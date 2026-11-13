from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC_NODE = "semantic_node"
    SEMANTIC_EDGE = "semantic_edge"


class MemoryEpisode(BaseModel):
    """Represents a discrete experiential event in the agent's episodic memory."""
    id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:10]}")
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    associated_entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mark_accessed(self, timestamp: Optional[datetime] = None) -> None:
        self.access_count += 1
        self.last_accessed_at = timestamp or datetime.now(timezone.utc)


class EntityNode(BaseModel):
    """Represents a discrete semantic entity concept in the knowledge graph."""
    id: str
    name: str
    entity_type: str = "CONCEPT"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    activation_energy: float = Field(default=0.0, ge=0.0)
    mention_count: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    """Represents a directed associative relationship between two concepts."""
    source_id: str
    target_id: str
    relation: str = "RELATED_TO"
    weight: float = Field(default=1.0, ge=0.0, le=5.0)
    co_occurrence_count: int = 1
    last_reinforced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def edge_key(self) -> str:
        return f"{self.source_id}->{self.relation}->{self.target_id}"


class RetrievalResult(BaseModel):
    """Result of an associative multi-hop memory recall."""
    episodes: List[MemoryEpisode] = Field(default_factory=list)
    activated_entities: List[EntityNode] = Field(default_factory=list)
    association_paths: List[List[str]] = Field(default_factory=list)
    query: str
    recall_latency_ms: float = 0.0

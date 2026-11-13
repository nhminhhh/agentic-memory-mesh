from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from memory_mesh.core.types import EntityNode, MemoryEpisode, RelationEdge
from memory_mesh.graph.store import KnowledgeGraphStore


class SQLiteMemoryStore:
    """Thread-safe SQLite persistent backing for episodes and knowledge graph topology."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    associated_entities TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    mention_count INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_key TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL,
                    co_occurrence_count INTEGER NOT NULL,
                    last_reinforced_at TEXT NOT NULL
                );
                """
            )

    def save_episode(self, episode: MemoryEpisode) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes 
                (id, content, importance, access_count, created_at, last_accessed_at, associated_entities, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.content,
                    episode.importance,
                    episode.access_count,
                    episode.created_at.isoformat(),
                    episode.last_accessed_at.isoformat(),
                    json.dumps(episode.associated_entities),
                    json.dumps(episode.metadata),
                ),
            )

    def load_episodes(self) -> List[MemoryEpisode]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM episodes ORDER BY created_at DESC")
            episodes: List[MemoryEpisode] = []
            for row in cursor.fetchall():
                episodes.append(
                    MemoryEpisode(
                        id=row["id"],
                        content=row["content"],
                        importance=row["importance"],
                        access_count=row["access_count"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
                        associated_entities=json.loads(row["associated_entities"]),
                        metadata=json.loads(row["metadata"]),
                    )
                )
            return episodes

    def save_graph(self, graph: KnowledgeGraphStore) -> None:
        with self._get_connection() as conn:
            for node in graph.nodes.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO graph_nodes (id, name, entity_type, importance, mention_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.name,
                        node.entity_type,
                        node.importance,
                        node.mention_count,
                        json.dumps(node.metadata),
                    ),
                )
            for edge in graph.edges.values():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO graph_edges 
                    (edge_key, source_id, target_id, relation, weight, co_occurrence_count, last_reinforced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge.edge_key,
                        edge.source_id,
                        edge.target_id,
                        edge.relation,
                        edge.weight,
                        edge.co_occurrence_count,
                        edge.last_reinforced_at.isoformat(),
                    ),
                )

    def load_graph(self) -> KnowledgeGraphStore:
        graph = KnowledgeGraphStore()
        with self._get_connection() as conn:
            for row in conn.execute("SELECT * FROM graph_nodes").fetchall():
                node = EntityNode(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    importance=row["importance"],
                    mention_count=row["mention_count"],
                    metadata=json.loads(row["metadata"]),
                )
                graph.nodes[node.id] = node
                graph._adjacency[node.id] = {}
                graph._reverse_adjacency[node.id] = {}

            for row in conn.execute("SELECT * FROM graph_edges").fetchall():
                edge = RelationEdge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation=row["relation"],
                    weight=row["weight"],
                    co_occurrence_count=row["co_occurrence_count"],
                    last_reinforced_at=datetime.fromisoformat(row["last_reinforced_at"]),
                )
                graph.edges[edge.edge_key] = edge
                graph._adjacency.setdefault(edge.source_id, {})[edge.target_id] = edge.edge_key
                graph._reverse_adjacency.setdefault(edge.target_id, {})[edge.source_id] = edge.edge_key

        return graph

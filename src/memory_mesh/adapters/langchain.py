from __future__ import annotations

from typing import Any, Dict, List, Optional
from memory_mesh.engine import MemoryMeshEngine


class MemoryMeshChatMessageHistory:
    """
    Adapter allowing MemoryMesh to act as drop-in long-term conversational memory
    for LangChain, LlamaIndex, or custom Agent frameworks.
    """

    def __init__(self, engine: Optional[MemoryMeshEngine] = None, session_id: str = "default_session"):
        self.engine = engine or MemoryMeshEngine()
        self.session_id = session_id
        self._message_log: List[Dict[str, str]] = []

    def add_user_message(
        self,
        message: str,
        importance: float = 0.6,
        entities: Optional[List[str]] = None,
    ) -> None:
        self._message_log.append({"role": "user", "content": message})
        self.engine.ingest(
            content=f"User: {message}",
            importance=importance,
            entities=entities,
            metadata={"role": "user", "session_id": self.session_id},
        )

    def add_ai_message(
        self,
        message: str,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
    ) -> None:
        self._message_log.append({"role": "assistant", "content": message})
        self.engine.ingest(
            content=f"Assistant: {message}",
            importance=importance,
            entities=entities,
            metadata={"role": "assistant", "session_id": self.session_id},
        )

    def load_memory_context(self, current_query: str, top_k: int = 3) -> str:
        """Retrieves and formats associative long-term memory for injection into agent prompt."""
        result = self.engine.recall(query=current_query, top_k=top_k)
        if not result.episodes:
            return ""

        context_lines = ["--- RELEVANT ASSOCIATIVE MEMORIES ---"]
        for ep in result.episodes:
            context_lines.append(f"• {ep.content}")

        if result.activated_entities:
            context_lines.append(
                f"[Active Concepts: {', '.join(n.name for n in result.activated_entities[:5])}]"
            )

        return "\n".join(context_lines)

    @property
    def messages(self) -> List[Dict[str, str]]:
        return list(self._message_log)

    def clear(self) -> None:
        self._message_log.clear()

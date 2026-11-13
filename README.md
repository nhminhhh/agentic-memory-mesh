# Agentic Memory Mesh

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Pydantic: v2](https://img.shields.io/badge/Pydantic-v2-orange.svg)](https://docs.pydantic.dev/)

**Agentic Memory Mesh** is a graph-augmented episodic and semantic long-term memory engine designed for autonomous LLM agents.

---

## 🧠 Why Memory Mesh?

Standard vector RAG provides flat semantic similarity over static text chunks, but fails when agents require:
1. **Multi-Hop Associative Reasoning**: Following associative links across conversation sessions (e.g. `Alice -> uses -> MacStudio -> runs -> Ollama`).
2. **Temporal Decay & Ebbinghaus Forgetting**: Naturally de-prioritizing stale observations while preserving frequently accessed or high-importance memories.
3. **Spreading Activation Retrieval**: Activating related concepts dynamically, mimicking cognitive associative recall.

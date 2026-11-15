from memory_mesh.engine import MemoryMeshEngine
from memory_mesh.reflection.synthesizer import CognitiveReflector


def test_cognitive_reflection_loop():
    engine = MemoryMeshEngine()

    engine.ingest("Minh is coding PyTorch neural networks", importance=0.8, entities=["Minh", "PyTorch"])
    engine.ingest("Minh is optimizing PyTorch CUDA kernels", importance=0.85, entities=["Minh", "PyTorch", "CUDA"])
    engine.ingest("Minh deployed PyTorch models to production", importance=0.8, entities=["Minh", "PyTorch"])

    # Trigger self-reflection
    reflections = engine.reflect()
    assert len(reflections) >= 1
    
    first = reflections[0]
    assert "Agent Insight" in first.content
    assert first.importance >= 0.85
    assert first.metadata["is_reflection"] is True
    assert len(first.metadata["evidence_episode_ids"]) >= 2

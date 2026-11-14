from memory_mesh.core.types import MemoryEpisode
from memory_mesh.compression.compactor import MemoryCompactor


def test_memory_compactor_merging():
    compactor = MemoryCompactor(similarity_threshold=0.35)

    ep1 = MemoryEpisode(content="Alice is deploying PyTorch models", importance=0.7, associated_entities=["Alice", "PyTorch"])
    ep2 = MemoryEpisode(content="Alice updated PyTorch inference pipeline", importance=0.9, associated_entities=["Alice", "PyTorch"])
    ep3 = MemoryEpisode(content="Bob is having lunch in Tokyo", importance=0.4, associated_entities=["Bob", "Tokyo"])

    compacted = compactor.compact([ep1, ep2, ep3])

    assert len(compacted) == 2
    # ep1 and ep2 should merge
    merged = next(e for e in compacted if "Consolidated Digest" in e.content)
    assert "Alice" in merged.associated_entities
    assert "PyTorch" in merged.associated_entities
    assert merged.importance == 0.9
    assert merged.metadata["compacted_from_count"] == 2

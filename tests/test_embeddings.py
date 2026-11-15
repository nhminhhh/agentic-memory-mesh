from memory_mesh.embeddings.dense import DenseSemanticModel, cosine_similarity
from memory_mesh.retrieval.ranker import MemoryRanker
from memory_mesh.core.types import MemoryEpisode


def test_dense_semantic_encoder():
    model = DenseSemanticModel(dimension=64)

    v1 = model.encode("Machine learning models with PyTorch")
    v2 = model.encode("Deep learning pipelines with PyTorch")
    v3 = model.encode("Baking chocolate chip cookies in the oven")

    sim_related = cosine_similarity(v1, v2)
    sim_unrelated = cosine_similarity(v1, v3)

    assert len(v1) == 64
    assert sim_related > sim_unrelated
    assert sim_related > 0.4


def test_hybrid_ranking():
    ranker = MemoryRanker()
    
    ep1 = MemoryEpisode(content="User develops PyTorch neural networks", associated_entities=["pytorch"])
    ep2 = MemoryEpisode(content="User likes gardening and plants in the backyard", associated_entities=["gardening"])

    ranked = ranker.rank_episodes([ep1, ep2], query="PyTorch network models", entity_activations={}, top_k=2)

    # ep1 should rank first due to dense semantic similarity
    assert ranked[0][0].id == ep1.id

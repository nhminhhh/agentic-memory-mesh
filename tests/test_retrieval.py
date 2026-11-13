from memory_mesh.core.types import MemoryEpisode
from memory_mesh.graph.store import KnowledgeGraphStore
from memory_mesh.retrieval.ranker import MemoryRanker
from memory_mesh.retrieval.spreading_activation import SpreadingActivationEngine


def test_spreading_activation_multi_hop():
    graph = KnowledgeGraphStore()
    
    # Linear chain: Alice -> uses -> NeoVim -> built_in -> Lua
    graph.add_relation("Alice", "NeoVim", relation="USES", weight=2.0)
    graph.add_relation("NeoVim", "Lua", relation="CONFIGURED_WITH", weight=1.5)

    engine = SpreadingActivationEngine(decay_factor=0.8, activation_threshold=0.05, max_hops=3)
    seed = [graph.upsert_entity("Alice")]

    energies, paths = engine.spread(graph, seed)

    assert "alice" in energies
    assert "neovim" in energies
    assert "lua" in energies

    # Energy should decay along hops: Alice > NeoVim > Lua
    assert energies["alice"] > energies["neovim"] > energies["lua"]
    assert len(paths) >= 2


def test_memory_ranker_composite_scoring():
    ranker = MemoryRanker()
    
    ep1 = MemoryEpisode(content="User prefers Lua scripts for NeoVim config", associated_entities=["neovim", "lua"])
    ep2 = MemoryEpisode(content="Discussed general weather in Hanoi", associated_entities=["hanoi"])

    entity_acts = {"neovim": 0.8, "lua": 0.6}
    query = "How to configure editor plugins?"

    ranked = ranker.rank_episodes([ep1, ep2], query=query, entity_activations=entity_acts, top_k=2)

    assert len(ranked) == 2
    # ep1 should rank first due to graph activation
    assert ranked[0][0].id == ep1.id
    assert ranked[0][1] > ranked[1][1]

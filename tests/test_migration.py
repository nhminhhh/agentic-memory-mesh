import tempfile
from pathlib import Path
from memory_mesh.core.types import MemoryEpisode
from memory_mesh.graph.store import KnowledgeGraphStore
from memory_mesh.storage.migration import MemoryMigrationToolkit


def test_migration_toolkit():
    graph = KnowledgeGraphStore()
    graph.add_relation("Minh", "PyTorch", relation="DEVELOPS", weight=2.0)
    graph.add_relation("PyTorch", "CUDA", relation="ACCELERATED_BY", weight=1.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Test Cypher export
        cql_file = Path(tmp_dir) / "graph.cql"
        cql = MemoryMigrationToolkit.export_neo4j_cypher(graph, cql_file)
        assert "MERGE (n:Entity {id: 'minh'})" in cql
        assert "MERGE (src)-[r:DEVELOPS" in cql
        assert cql_file.exists()

        # 2. Test Gephi GEXF export
        gexf_file = Path(tmp_dir) / "graph.gexf"
        gexf = MemoryMigrationToolkit.export_gephi_gexf(graph, gexf_file)
        assert "<gexf" in gexf
        assert 'source="minh"' in gexf
        assert gexf_file.exists()

        # 3. Test Fine-tuning JSONL export
        episodes = [MemoryEpisode(content="Minh runs PyTorch on CUDA", associated_entities=["Minh", "PyTorch"])]
        jsonl_file = Path(tmp_dir) / "dataset.jsonl"
        jsonl_str = MemoryMigrationToolkit.export_fine_tuning_jsonl(episodes, jsonl_file)
        assert "Minh runs PyTorch on CUDA" in jsonl_str
        assert jsonl_file.exists()

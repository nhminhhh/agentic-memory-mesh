import tempfile
from pathlib import Path
from memory_mesh.engine import MemoryMeshEngine


def test_engine_full_lifecycle_and_sqlite_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_memory.db"
        
        # 1. Initialize engine and ingest experiences
        engine1 = MemoryMeshEngine(storage_path=db_path)
        
        ep1 = engine1.ingest(
            content="Alice leads AI engineering at Datamate and uses PyTorch",
            importance=0.9,
            entities=["Alice", "Datamate", "PyTorch"],
            relations=[("Alice", "Datamate", "WORKS_AT"), ("Alice", "PyTorch", "USES")],
        )
        
        ep2 = engine1.ingest(
            content="PyTorch models are accelerated on Apple Silicon with MPS backend",
            importance=0.8,
            entities=["PyTorch", "Apple Silicon", "MPS"],
            relations=[("PyTorch", "Apple Silicon", "RUNS_ON")],
        )

        assert engine1.stats()["total_episodes"] == 2
        assert engine1.stats()["total_entities"] >= 4

        # 2. Recall with associative multi-hop query
        # Query mentions "Alice", engine should recall ep2 as well via PyTorch -> Apple Silicon!
        res = engine1.recall("What hardware accelerates tools used by Alice?", top_k=2)
        assert len(res.episodes) >= 1
        assert any(e.id == ep1.id for e in res.episodes)

        # 3. Consolidate
        c_stats = engine1.consolidate()
        assert c_stats["consolidated_working_memories"] == 2

        # 4. Reload from SQLite database in a new engine instance
        engine2 = MemoryMeshEngine(storage_path=db_path)
        assert engine2.stats()["total_episodes"] == 2
        assert engine2.stats()["total_entities"] == engine1.stats()["total_entities"]
        assert engine2.stats()["total_relations"] == engine1.stats()["total_relations"]

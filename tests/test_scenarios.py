

def test_associative_recall_scenario_2026_11_14_ebc501():
    from memory_mesh.engine import MemoryMeshEngine
    engine = MemoryMeshEngine()
    engine.ingest(
        content="Testing auto-linking between Entity_2026_11_14_ebc501 and Tool_ebc501",
        importance=0.8,
        entities=["Entity_2026_11_14_ebc501", "Tool_ebc501"],
        relations=[("Entity_2026_11_14_ebc501", "Tool_ebc501", "UTILIZES")],
    )
    res = engine.recall("Entity_2026_11_14_ebc501", top_k=1)
    assert len(res.episodes) == 1


def test_associative_recall_scenario_2026_11_15_907769():
    from memory_mesh.engine import MemoryMeshEngine
    engine = MemoryMeshEngine()
    engine.ingest(
        content="Testing auto-linking between Entity_2026_11_15_907769 and Tool_907769",
        importance=0.8,
        entities=["Entity_2026_11_15_907769", "Tool_907769"],
        relations=[("Entity_2026_11_15_907769", "Tool_907769", "UTILIZES")],
    )
    res = engine.recall("Entity_2026_11_15_907769", top_k=1)
    assert len(res.episodes) == 1

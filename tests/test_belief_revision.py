from memory_mesh.engine import MemoryMeshEngine


def test_belief_revision_superseding():
    engine = MemoryMeshEngine()

    # Step 1: Initial belief
    ep1 = engine.ingest(
        content="Minh lives in Hanoi and studies AI",
        importance=0.9,
        entities=["Minh", "Hanoi"],
        relations=[("Minh", "Hanoi", "LIVES_IN")],
    )

    # Verify edge exists
    neighbors_1 = [n.id for n, _ in engine.graph.get_neighbors("minh", bidirectional=False)]
    assert "hanoi" in neighbors_1

    # Step 2: User updates state with contradictory relocation
    ep2 = engine.ingest(
        content="Minh moved and now lives in Da Nang",
        importance=0.9,
        entities=["Minh", "Da Nang"],
        relations=[("Minh", "Da Nang", "LIVES_IN")],
    )

    # Outgoing LIVES_IN from Minh should now be 'da_nang', NOT 'hanoi'
    outgoing_edges = engine.graph.get_neighbors("minh", bidirectional=False)
    lives_in_targets = [edge.target_id for _, edge in outgoing_edges if edge.relation == "LIVES_IN"]
    assert "da_nang" in lives_in_targets
    assert "hanoi" not in lives_in_targets

    # Old episode should be soft-deprecated (importance downweighted)
    assert ep1.importance < 0.5
    assert ep1.metadata.get("superseded_by") == "Minh->LIVES_IN->Da Nang"

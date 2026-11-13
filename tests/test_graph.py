from memory_mesh.graph.store import KnowledgeGraphStore


def test_knowledge_graph_upsert_and_relations():
    graph = KnowledgeGraphStore()
    
    n_alice = graph.upsert_entity("Alice", entity_type="PERSON", importance=0.9)
    n_py = graph.upsert_entity("Python", entity_type="TECH", importance=0.8)
    n_fastapi = graph.upsert_entity("FastAPI", entity_type="TECH", importance=0.7)

    assert n_alice.id == "alice"
    assert n_py.id == "python"

    # Add relations
    e1 = graph.add_relation("Alice", "Python", relation="PROGRAMS_IN", weight=1.5)
    e2 = graph.add_relation("FastAPI", "Python", relation="BUILT_ON", weight=2.0)

    assert e1.weight == 1.5
    # Reinforce relation
    e1_again = graph.add_relation("Alice", "Python", relation="PROGRAMS_IN")
    assert e1_again.weight > 1.5
    assert e1_again.co_occurrence_count == 2

    # Query neighbors
    neighbors = graph.get_neighbors("alice")
    neighbor_ids = [n.id for n, _ in neighbors]
    assert "python" in neighbor_ids


def test_find_entities_in_text():
    graph = KnowledgeGraphStore()
    graph.upsert_entity("Docker", entity_type="TECH")
    graph.upsert_entity("PostgreSQL", entity_type="DATABASE")

    text = "We are deploying PostgreSQL inside Docker containers on AWS."
    found = graph.find_entities_in_text(text)
    found_ids = {n.id for n in found}

    assert "docker" in found_ids
    assert "postgresql" in found_ids

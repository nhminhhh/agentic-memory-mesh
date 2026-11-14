import tempfile
from pathlib import Path
from memory_mesh.graph.store import KnowledgeGraphStore
from memory_mesh.graph.visualizer import GraphVisualizer


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


def test_graph_visualizer():
    graph = KnowledgeGraphStore()
    graph.add_relation("Alice", "Bob", relation="COLLABORATES_WITH")
    
    mermaid_str = GraphVisualizer.to_mermaid(graph)
    assert "flowchart LR" in mermaid_str
    assert "Alice" in mermaid_str
    assert "COLLABORATES_WITH" in mermaid_str

    with tempfile.TemporaryDirectory() as tmp_dir:
        html_file = Path(tmp_dir) / "graph.html"
        GraphVisualizer.to_interactive_html(graph, html_file)
        assert html_file.exists()
        assert "vis.Network" in html_file.read_text(encoding="utf-8")

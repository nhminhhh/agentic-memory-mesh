from memory_mesh.extraction.extractor import EntityRelationExtractor
from memory_mesh.engine import MemoryMeshEngine


def test_entity_relation_extractor():
    text = "Minh develops VLM applications. Minh uses PyTorch on Docker containers."
    
    entities = EntityRelationExtractor.extract_entities(text)
    assert "Minh" in entities
    assert "Pytorch" in entities or "PyTorch" in entities
    assert "Docker" in entities

    relations = EntityRelationExtractor.extract_relations(text)
    assert len(relations) >= 2
    relations_dict = {(r[0], r[1]): r[2] for r in relations}
    assert relations_dict.get(("Minh", "Vlm")) == "DEVELOPS"
    assert relations_dict.get(("Minh", "Pytorch")) == "USES" or relations_dict.get(("Minh", "PyTorch")) == "USES"


def test_auto_extraction_in_engine():
    engine = MemoryMeshEngine()
    ep = engine.ingest("Alice works at Google and uses Kubernetes daily")
    
    assert len(ep.associated_entities) >= 2
    assert "Alice" in ep.associated_entities
    # Graph should have automatically linked Alice to Google
    neighbors = engine.graph.get_neighbors("alice")
    neighbor_names = [n.name.lower() for n, _ in neighbors]
    assert "google" in neighbor_names or "kubernetes" in neighbor_names

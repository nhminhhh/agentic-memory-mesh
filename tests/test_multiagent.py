from memory_mesh.multiagent.namespaces import MultiAgentMemoryHub


def test_multi_agent_namespace_isolation_and_sharing():
    hub = MultiAgentMemoryHub()

    # Agent 1 (Researcher) ingests private observation
    ep_r = hub.ingest_private("researcher", "Observed anomalous training loss in checkpoint 42", importance=0.8)
    
    # Agent 2 (Coder) ingests private observation
    ep_c = hub.ingest_private("coder", "Refactored CUDA kernels in attention block", importance=0.7)

    # Ingest shared team convention
    hub.ingest_shared("lead", "Codebase uses PyTorch 2.5 with FlashAttention-3", importance=0.9, entities=["PyTorch", "FlashAttention"])

    # 1. Coder should NOT see Researcher's private thoughts
    coder_recall = hub.recall_blended("coder", "training loss checkpoint", top_k=3)
    assert not any("anomalous training loss" in ep.content for ep in coder_recall.episodes)

    # 2. Both agents SHOULD see shared conventions
    coder_shared_recall = hub.recall_blended("coder", "PyTorch FlashAttention", top_k=3)
    assert any("PyTorch 2.5" in ep.content for ep in coder_shared_recall.episodes)

    # 3. Promote Researcher's finding to shared
    promoted = hub.promote_to_shared("researcher", ep_r.id)
    assert promoted is not None

    # Now Coder CAN see the promoted discovery
    coder_recall_after = hub.recall_blended("coder", "training loss checkpoint", top_k=3)
    assert any("anomalous training loss" in ep.content for ep in coder_recall_after.episodes)

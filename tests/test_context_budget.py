from memory_mesh.context.budget_synthesizer import ContextBudgetSynthesizer
from memory_mesh.core.types import EntityNode, MemoryEpisode, RetrievalResult


def test_context_budget_synthesizer():
    synthesizer = ContextBudgetSynthesizer()

    episodes = [
        MemoryEpisode(content="Episode 1: Minh is training vision-language models on Nvidia A100 GPUs"),
        MemoryEpisode(content="Episode 2: The pipeline uses FlashAttention-3 to achieve 40% speedup"),
        MemoryEpisode(content="Episode 3: Batch size was configured to 64 with gradient accumulation"),
        MemoryEpisode(content="Episode 4: Checkpoints are synchronized to cloud storage every 1000 steps"),
    ]
    concepts = [EntityNode(id="vlm", name="VLM"), EntityNode(id="a100", name="A100")]
    
    retrieval = RetrievalResult(
        query="Hardware setup",
        episodes=episodes,
        activated_entities=concepts,
        association_paths=[["Minh", "USES", "A100"]],
    )

    # Pack with large budget (all fit)
    packed_large = synthesizer.pack(retrieval, max_tokens=500)
    assert packed_large.included_episodes_count == 4
    assert packed_large.dropped_episodes_count == 0

    # Pack with very strict budget (e.g. 50 tokens) -> should drop lower priority episodes
    packed_small = synthesizer.pack(retrieval, max_tokens=60)
    assert packed_small.estimated_tokens <= 60
    assert packed_small.dropped_episodes_count > 0
    assert "CONTEXTUAL LONG-TERM MEMORY" in packed_small.formatted_prompt

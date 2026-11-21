"""
Multi-Agent Swarm Memory & Belief Revision Demo.
Demonstrates:
  1. Private vs Shared Swarm Memory with MultiAgentMemoryHub
  2. Automatic Contradiction Resolution & Belief Revision
  3. Context Budget Prompt Synthesis within strict token bounds
"""

from memory_mesh.context.budget_synthesizer import ContextBudgetSynthesizer
from memory_mesh.multiagent.namespaces import MultiAgentMemoryHub


def main():
    print("=" * 70)
    print(" Agentic Memory Mesh - Multi-Agent Swarm & Belief Revision Demo")
    print("=" * 70)

    hub = MultiAgentMemoryHub()

    print("\n[Step 1] Initializing Multi-Agent Swarm Memories...")
    # Lead Agent establishes shared knowledge
    hub.ingest_shared(
        author_agent_id="lead_agent",
        content="Project goal: Train vision-language models for document intelligence",
        importance=0.9,
        entities=["Minh", "VLM", "Document Intelligence"],
    )

    # Researcher Agent adds private scratchpad thoughts
    ep_r1 = hub.ingest_private(
        agent_id="researcher",
        content="Hypothesis: Swapping OCR backbone to PaddleOCR will boost table parsing accuracy",
        importance=0.8,
        entities=["PaddleOCR", "Table Parsing"],
    )
    print(" ✓ Lead established shared goals.")
    print(" ✓ Researcher recorded private hypothesis.")

    print("\n[Step 2] Promoting Researcher Discovery to Collective Swarm Memory...")
    hub.promote_to_shared("researcher", ep_r1.id)
    print(" ✓ Researcher discovery promoted to collective knowledge.")

    print("\n[Step 3] Handling State Evolution with Belief Revision...")
    lead_engine = hub.get_agent_engine("lead_agent")

    # Day 1 Fact
    lead_engine.ingest(
        content="Minh lives in Hanoi and leads the engineering team",
        importance=0.9,
        entities=["Minh", "Hanoi"],
        relations=[("Minh", "Hanoi", "LIVES_IN")],
    )
    print(" Day 1: Recorded that Minh lives in Hanoi.")

    # Day 10 Fact (Contradictory relocation)
    lead_engine.ingest(
        content="Minh relocated and now lives in Da Nang",
        importance=0.9,
        entities=["Minh", "Da Nang"],
        relations=[("Minh", "Da Nang", "LIVES_IN")],
    )
    print(" Day 10: Minh moved to Da Nang -> Belief Revision triggered!")

    # Verify Da Nang is active while Hanoi is superseded
    neighbors = [edge.target_id for _, edge in lead_engine.graph.get_neighbors("minh", bidirectional=False) if edge.relation == "LIVES_IN"]
    print(f" Current Active 'LIVES_IN' target: {neighbors} (Hanoi archived)")

    print("\n[Step 4] Packing Memories into Strict Context Token Budget...")
    recall_result = hub.recall_blended("coder", "What is Minh working on and where is he located?", top_k=4)
    synthesizer = ContextBudgetSynthesizer()
    packed = synthesizer.pack(recall_result, max_tokens=120)

    print(f"\nPacked Prompt Context ({packed.estimated_tokens}/{packed.budget_limit_tokens} tokens):")
    print("-" * 50)
    print(packed.formatted_prompt)
    print("-" * 50)
    print(f"Included: {packed.included_episodes_count} episodes | Dropped: {packed.dropped_episodes_count} lower-priority items")


if __name__ == "__main__":
    main()

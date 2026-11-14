from memory_mesh.adapters.langchain import MemoryMeshChatMessageHistory


def test_chat_message_history_adapter():
    history = MemoryMeshChatMessageHistory(session_id="session_42")
    
    history.add_user_message("I am building an autonomous agent using PyTorch", importance=0.8, entities=["PyTorch"])
    history.add_ai_message("Great choice! PyTorch gives you complete flexibility.", importance=0.5)

    assert len(history.messages) == 2
    assert history.messages[0]["role"] == "user"
    assert history.messages[1]["role"] == "assistant"

    # Context retrieval for next turn
    context = history.load_memory_context("What library was I using for my agent?")
    assert "PyTorch" in context
    assert "RELEVANT ASSOCIATIVE MEMORIES" in context

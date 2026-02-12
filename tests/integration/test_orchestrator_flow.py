import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.orchestrator import AdminOrchestrator
from langchain_core.messages import AIMessage
from src.agents.intent_classifier import intent_classifier


@pytest.mark.asyncio
async def test_orchestrator_full_flow_simple_qa():
    # Mocks
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="Generated Answer")

    mock_retriever = AsyncMock(return_value=[{"source": "S1", "content": "C1"}])
    mock_translator = AsyncMock(
        side_effect=lambda text, **kwargs: text
    )  # No-op translator

    # Mock Redis (Cache)
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None

    # Mock Memory Redis
    mock_mem_client = AsyncMock()
    mock_mem_client.get.return_value = None  # No existing state

    with patch("src.agents.orchestrator.ChatOpenAI", return_value=mock_llm), patch(
        "src.agents.orchestrator.retrieve_legal_info", mock_retriever
    ), patch("src.agents.orchestrator.translate_admin_text", mock_translator), patch(
        "src.agents.orchestrator.redis.Redis", return_value=mock_cache
    ), patch.object(
        intent_classifier, "classify", new_callable=AsyncMock
    ) as mock_classify, patch(
        "src.memory.manager.memory_manager.redis_client", mock_mem_client
    ), patch(
        "src.shared.guardrails.guardrail_manager.validate_topic", new_callable=AsyncMock
    ) as mock_validate, patch(
        "src.shared.guardrails.guardrail_manager.check_hallucination",
        new_callable=AsyncMock,
    ) as mock_hallucination, patch(
        "src.shared.guardrails.guardrail_manager.add_disclaimer",
        side_effect=lambda x, y: x,
    ), patch("src.memory.manager.RedisChatMessageHistory") as mock_redis_history:
        # Mock Legacy History
        mock_history_instance = MagicMock()
        mock_history_instance.messages = []
        mock_redis_history.return_value = mock_history_instance
        mock_classify.return_value = "SIMPLE_QA"
        mock_validate.return_value = (True, "Valid")
        mock_hallucination.return_value = True  # No hallucination

        mock_classify.return_value = "SIMPLE_QA"

        # Initialize
        orch = AdminOrchestrator()

        # Run
        session_id = "test_sess_flow"
        response = await orch.handle_query("How to get a visa?", session_id=session_id)

        # Verify Response
        assert response == "Generated Answer"

        # Verify Intent Classification execution
        mock_classify.assert_called_once()

        # Verify State Persistence
        # memory_manager.redis_client.set should be called to save the state
        assert mock_mem_client.set.called

        # Inspect what was saved
        args, _ = mock_mem_client.set.call_args
        key, val = args
        assert key == f"agent_state:{session_id}"
        assert "SIMPLE_QA" in val  # Validates intent was saved
        assert "Generated Answer" in val  # Validates message history saved

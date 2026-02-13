import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import AdminOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test that orchestrator initializes correctly."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"),
    ):
        orchestrator = AdminOrchestrator()
        assert orchestrator.llm is not None
        assert orchestrator.cache is not None


@pytest.mark.asyncio
async def test_handle_query_cache_hit():
    """Test that cache returns value if present."""
    with patch("src.agents.orchestrator.redis.Redis") as mock_redis_cls:
        # Mock Redis instance
        mock_redis = AsyncMock()
        mock_redis_cls.return_value = mock_redis

        # Setup cache hit
        mock_redis.get.return_value = "Cached Response"

        orchestrator = AdminOrchestrator()
        # Inject our mock instance (since __init__ creates a new one from the class mock)
        orchestrator.cache = mock_redis

        # Ensure DEBUG is False for cache to work
        with (
            patch("src.config.settings.DEBUG", False),
            patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"),
        ):
            orchestrator = AdminOrchestrator()
            # Inject our mock instance (since __init__ creates a new one from the class mock)
            orchestrator.cache = mock_redis

            response = await orchestrator.handle_query("test query", "fr")
            assert response == "Cached Response"
            mock_redis.get.assert_called_once()


@pytest.mark.asyncio
async def test_handle_query_flow():
    """Test the full flow with mocks."""
    from src.agents.state import AgentState

    with (
        patch("src.agents.orchestrator.redis.Redis") as mock_redis_cls,
        patch(
            "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retriever,
        patch(
            "src.agents.orchestrator.translate_admin_text", new_callable=AsyncMock
        ) as _,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch(
            "src.shared.guardrails.guardrail_manager.check_hallucination",
            new_callable=AsyncMock,
        ) as mock_hallucination,
        patch(
            "src.shared.guardrails.guardrail_manager.add_disclaimer",
            side_effect=lambda x, y: x,
        ),
        patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"),
        patch("src.agents.orchestrator.memory_manager") as mock_memory_manager,
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
        ) as mock_classify,
    ):
        # Setup Mocks
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Cache miss
        mock_redis_cls.return_value = mock_redis

        orchestrator = AdminOrchestrator()
        orchestrator.cache = mock_redis
        orchestrator.llm = MagicMock()
        orchestrator.llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Generated Answer")
        )

        # Setup Mock Memory with AgentState
        mock_state = AgentState(session_id="test_session", messages=[])
        mock_memory_manager.load_agent_state = AsyncMock(return_value=mock_state)
        mock_memory_manager.save_agent_state = AsyncMock()

        # Setup Intent Classifier
        mock_classify.return_value = "SIMPLE_QA"

        mock_validate.return_value = (True, "")
        mock_retriever.return_value = [
            {"source": "test", "content": "context", "metadata": {}}
        ]
        mock_hallucination.return_value = True

        # Execute
        response = await orchestrator.handle_query(
            "Valid query", session_id="test_session"
        )

        # Verify
        mock_validate.assert_called_once()
        mock_retriever.assert_called_once()
        orchestrator.llm.ainvoke.assert_called_once()
        mock_classify.assert_called_once()
        mock_memory_manager.load_agent_state.assert_called_once()
        mock_memory_manager.save_agent_state.assert_called_once()

        assert "Generated Answer" in response


@pytest.mark.asyncio
async def test_orchestrator_cache_exceptions():
    from src.agents.state import AgentState
    from src.agents.intent_classifier import Intent

    with patch("src.agents.orchestrator.redis.Redis"):
        orchestrator = AdminOrchestrator()
        orchestrator.cache.get = AsyncMock(side_effect=Exception("Redis Down"))
        orchestrator.cache.setex = AsyncMock(side_effect=Exception("Redis Write Error"))

        # Should persist despite redis error
        with (
            patch(
                "src.agents.intent_classifier.intent_classifier.classify",
                new_callable=AsyncMock,
            ) as mock_classify,
            patch(
                "src.shared.guardrails.guardrail_manager.validate_topic",
                new_callable=AsyncMock,
            ) as mock_validate,
            patch(
                "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
            ),
            patch(
                "src.shared.guardrails.guardrail_manager.check_hallucination",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "src.agents.orchestrator.memory_manager.load_agent_state",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "src.agents.orchestrator.memory_manager.save_agent_state",
                new_callable=AsyncMock,
            ),
        ):
            mock_classify.return_value = Intent.SIMPLE_QA
            mock_validate.return_value = (True, "")
            mock_load.return_value = AgentState(session_id="test", messages=[])
            orchestrator.llm = MagicMock()
            orchestrator.llm.ainvoke = AsyncMock(return_value=MagicMock(content="Ans"))

            res = await orchestrator.handle_query("query")
            assert "Ans" in res


@pytest.mark.asyncio
async def test_orchestrator_guardrail_rejections():
    from src.agents.state import AgentState
    from src.agents.intent_classifier import Intent

    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
        ) as mock_classify,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch(
            "src.agents.orchestrator.memory_manager.load_agent_state",
            new_callable=AsyncMock,
        ) as mock_load,
        patch(
            "src.agents.orchestrator.memory_manager.save_agent_state",
            new_callable=AsyncMock,
        ),
    ):
        orchestrator = AdminOrchestrator()
        mock_load.return_value = AgentState(session_id="test", messages=[])

        # Test Topic Rejection
        mock_classify.return_value = Intent.SIMPLE_QA
        mock_validate.return_value = (False, "Off topic")

        res = await orchestrator.handle_query("bad query")
        assert "Désolé" in res
        assert "Off topic" in res


@pytest.mark.asyncio
async def test_orchestrator_agent_graph_hallucination():
    """Test that Orchestrator catches hallucinations from AgentGraph."""
    from src.agents.state import AgentState
    from src.agents.intent_classifier import Intent
    from langchain_core.messages import AIMessage

    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
        ) as mock_classify,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch(
            "src.shared.guardrails.guardrail_manager.check_hallucination",
            new_callable=AsyncMock,
        ) as mock_hallucination,
        patch(
            "src.agents.orchestrator.memory_manager.load_agent_state",
            new_callable=AsyncMock,
        ) as mock_load,
        patch(
            "src.agents.orchestrator.memory_manager.save_agent_state",
            new_callable=AsyncMock,
        ),
        patch(
            "src.agents.graph.agent_graph.ainvoke", new_callable=AsyncMock
        ) as mock_graph,
        patch(
            "src.agents.orchestrator.translate_admin_text",
            side_effect=lambda text, target_language: text,
        ),
        patch(
            "src.shared.guardrails.guardrail_manager.add_disclaimer",
            side_effect=lambda x, y: x,
        ),
    ):
        orchestrator = AdminOrchestrator()

        # Setup State
        state = AgentState(session_id="test", messages=[])
        mock_load.return_value = state

        # Setup Intent -> Complex to trigger Graph
        mock_classify.return_value = Intent.LEGAL_INQUIRY
        mock_validate.return_value = (True, "")

        # Setup Graph Return
        final_state = {"messages": [AIMessage(content="Hallucinated Answer")]}
        mock_graph.return_value = final_state

        # Setup Hallucination Check -> False (Hallucination Detected)
        mock_hallucination.return_value = False

        # Execute
        res = await orchestrator.handle_query("complex query")

        # Verify Fallback
        assert "Attention" in res or "Warning" in res
        mock_graph.assert_called()
        mock_hallucination.assert_called()

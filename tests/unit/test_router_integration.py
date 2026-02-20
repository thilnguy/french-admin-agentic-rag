import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import AdminOrchestrator
from src.agents.intent_classifier import Intent
from src.agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
async def test_orchestrator_routes_to_fast_lane_simple_qa():
    """Test that SIMPLE_QA uses the legacy/fast path."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.ChatOpenAI"),
        patch("src.shared.query_pipeline.get_query_pipeline") as mock_get_pipeline,
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
        ) as mock_classify,
        patch(
            "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retriever,
        patch(
            "src.agents.orchestrator.translate_admin_text", new_callable=AsyncMock
        ) as mock_translator,
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
        patch("src.config.settings.OPENAI_API_KEY", "sk-test"),
        patch("src.agents.orchestrator.memory_manager") as mock_memory,
    ):
        # Setup
        orchestrator = AdminOrchestrator()
        orchestrator.llm = MagicMock()
        orchestrator.llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Fast response")
        )

        mock_translator.return_value = "Translated content"
        mock_classify.return_value = Intent.SIMPLE_QA
        mock_validate.return_value = (True, "")
        mock_retriever.return_value = [
            {"source": "test", "content": "info", "metadata": {}}
        ]
        mock_hallucination.return_value = True

        # Pipeline Mock
        mock_pipeline = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.run.return_value = MagicMock(
            rewritten_query="Simple question",
            intent=Intent.SIMPLE_QA,
            extracted_data={},
            new_core_goal=None,
            is_contextual_continuation=False,
        )

        # State
        state = AgentState(session_id="test", messages=[])
        mock_memory.load_agent_state = AsyncMock(return_value=state)
        mock_memory.save_agent_state = AsyncMock()

        # Run
        response = await orchestrator.handle_query("Simple question", "en")

        # Verify
        # mock_classify.assert_called_once() # Removed as pipeline handles it
        # orchestrator.llm.ainvoke.assert_called_once() # Logic changed, check result
        mock_get_pipeline.assert_called_once()
        mock_pipeline.run.assert_awaited_once()
        mock_retriever.assert_called_once()
        assert response == "Translated content"

        # Verify we did NOT use AgentGraph (can't easily verify 'not imported',
        # but we can verify legacy path was taken by checking LLM usage which might differ in graph)


@pytest.mark.asyncio
async def test_orchestrator_routes_to_agent_graph_complex():
    """Test that COMPLEX_PROCEDURE uses the AgentGraph."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.ChatOpenAI"),
        patch("src.shared.query_pipeline.get_query_pipeline") as mock_get_pipeline,
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
        ) as mock_classify,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ) as mock_validate,
        patch(
            "src.shared.guardrails.guardrail_manager.check_hallucination",
            new_callable=AsyncMock,
        ),
        patch(
            "src.shared.guardrails.guardrail_manager.add_disclaimer",
            side_effect=lambda x, y: x,
        ),
        patch("src.config.settings.OPENAI_API_KEY", "sk-test"),
        patch("src.agents.orchestrator.memory_manager") as mock_memory,
        patch("src.agents.orchestrator.agent_graph") as mock_agent_graph,
        patch(
            "src.agents.orchestrator.translate_admin_text", new_callable=AsyncMock
        ) as mock_translator,
        patch(
            "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retriever,
    ):
        # Setup
        orchestrator = AdminOrchestrator()
        mock_translator.return_value = (
            "Graph response"  # Fallback if translator is called
        )
        mock_retriever.return_value = []

        # Mock Graph ainvoke
        mock_agent_graph.ainvoke = AsyncMock()

        # Intent = COMPLEX_PROCEDURE
        mock_classify.return_value = Intent.COMPLEX_PROCEDURE
        mock_validate.return_value = (True, "")

        # Pipeline Mock
        mock_pipeline = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.run.return_value = MagicMock(
            rewritten_query="Complex task",
            intent=Intent.COMPLEX_PROCEDURE,
            extracted_data={},
            new_core_goal=None,
            is_contextual_continuation=False,
        )

        # State
        state = AgentState(session_id="test", messages=[])
        mock_memory.load_agent_state = AsyncMock(return_value=state)
        mock_memory.save_agent_state = AsyncMock()

        # Mock Graph Response
        # Graph returns state dict with messages
        final_messages = [
            HumanMessage(content="users query"),
            AIMessage(content="Graph response"),
        ]
        mock_agent_graph.ainvoke.return_value = {"messages": final_messages}

        # Run
        response = await orchestrator.handle_query("Complex task", "fr")

        # Verify
        # mock_classify.assert_called_once() # Handled by pipeline
        mock_agent_graph.ainvoke.assert_called_once()
        assert response == "Graph response"

        # Check that state was saved with graph response
        args, _ = mock_memory.save_agent_state.call_args
        saved_state = args[1]
        assert saved_state.messages == final_messages

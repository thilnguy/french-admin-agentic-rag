import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import AdminOrchestrator
from src.agents.intent_classifier import Intent
from src.agents.state import AgentState
from src.shared.query_pipeline import PipelineResult


@pytest.mark.asyncio
async def test_stream_query_fast_lane():
    """Test streaming for SIMPLE_QA (Fast Lane)."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.shared.query_pipeline.get_query_pipeline") as mock_get_pipeline,
        patch(
            "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retriever,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch("src.agents.orchestrator.memory_manager") as mock_memory,
        patch("src.config.settings.DEBUG", False),
    ):
        # Setup
        orchestrator = AdminOrchestrator()
        orchestrator.cache = AsyncMock()
        orchestrator.cache.get.return_value = None

        # Mock Pipeline
        mock_pipeline_instance = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.return_value = PipelineResult(
            rewritten_query="Rewritten Hello",
            intent=Intent.SIMPLE_QA,
            extracted_data={},
            new_core_goal=None,
        )

        orchestrator.llm = MagicMock()

        # Mock LLM streaming
        async def mock_astream(messages):
            yield MagicMock(content="Hello")
            yield MagicMock(content=" World")

        orchestrator.llm.astream = mock_astream

        mock_validate.return_value = (True, "")
        mock_retriever.return_value = [{"source": "doc", "content": "info"}]

        state = AgentState(session_id="test", messages=[])
        mock_memory.load_agent_state = AsyncMock(return_value=state)
        mock_memory.save_agent_state = AsyncMock()

        # Execute
        events = []
        async for event in orchestrator.stream_query("Hello", "en"):
            events.append(event)

        # Verify
        assert len(events) > 0
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "Hello" in tokens
        assert " World" in tokens

        # Verify status messages
        statuses = [e["content"] for e in events if e["type"] == "status"]
        assert "Analysing request..." in statuses
        assert "Searching administrative database..." in statuses


@pytest.mark.asyncio
async def test_stream_query_slow_lane():
    """Test streaming for COMPLEX_PROCEDURE (Slow Lane/AgentGraph)."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.shared.query_pipeline.get_query_pipeline") as mock_get_pipeline,
        patch(
            "src.shared.guardrails.guardrail_manager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch(
            "src.agents.graph.agent_graph.astream_events",
            new_callable=MagicMock,
        ) as mock_graph_stream,
        patch("src.agents.orchestrator.memory_manager") as mock_memory,
        patch("src.config.settings.DEBUG", False),
    ):
        # Setup
        orchestrator = AdminOrchestrator()
        orchestrator.cache = AsyncMock()
        orchestrator.cache.get.return_value = None

        # Mock Pipeline
        mock_pipeline_instance = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run.return_value = PipelineResult(
            rewritten_query="Rewritten Complex",
            intent=Intent.COMPLEX_PROCEDURE,
            extracted_data={},
            new_core_goal=None,
        )

        mock_validate.return_value = (True, "")

        state = AgentState(session_id="test", messages=[])
        mock_memory.load_agent_state = AsyncMock(return_value=state)
        mock_memory.save_agent_state = AsyncMock()

        # Mock Graph Events
        async def event_generator(state, version):
            yield {"event": "on_tool_start", "name": "check_eligibility"}
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content="Eligible")},
            }
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": MagicMock(content="!")},
            }

        mock_graph_stream.side_effect = event_generator

        # Execute
        events = []
        async for event in orchestrator.stream_query("Combine steps", "fr"):
            events.append(event)

        # Verify
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "Eligible" in tokens
        assert "!" in tokens

        statuses = [e["content"] for e in events if e["type"] == "status"]
        assert "Routing to Expert Agent..." in statuses
        assert "Executing tool: check_eligibility..." in statuses


@pytest.mark.asyncio
async def test_stream_query_cache_hit():
    """Test streaming returns cached response."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.config.settings.DEBUG", False),
    ):
        orchestrator = AdminOrchestrator()
        orchestrator.cache = AsyncMock()
        orchestrator.cache.get.return_value = "Cached Answer"

        events = []
        async for event in orchestrator.stream_query("Cached query", "fr"):
            events.append(event)

        assert len(events) == 2
        assert events[0] == {"type": "status", "content": "Cache hit"}
        assert events[1] == {"type": "token", "content": "Cached Answer"}

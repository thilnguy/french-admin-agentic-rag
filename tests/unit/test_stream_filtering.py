import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import AgentState
from src.agents.intent_classifier import Intent


async def test_stream_filtering():
    print("Testing Stream Filtering...")

    # Mock dependencies
    mock_memory = AsyncMock()
    mock_memory.load_agent_state.return_value = AgentState(session_id="test")

    # Mock Graph
    mock_graph = AsyncMock()

    # Simulate a stream of events from the graph
    async def mock_astream_events(*args, **kwargs):
        events = [
            # Event 1: Internal thinking (Should be filtered)
            {
                "event": "on_chat_model_stream",
                "tags": ["internal"],
                "data": {"chunk": MagicMock(content="Thinking...")},
            },
            # Event 2: Real answer (Should be yielded)
            {
                "event": "on_chat_model_stream",
                "tags": [],
                "data": {"chunk": MagicMock(content="Hello user.")},
            },
        ]
        for e in events:
            yield e

    mock_graph.astream_events = mock_astream_events

    # Create Orchestrator
    with (
        patch("src.agents.orchestrator.memory_manager", mock_memory),
        patch("src.agents.graph.agent_graph", mock_graph),
        patch("src.agents.preprocessor.query_rewriter") as mock_rewriter,
        patch("src.agents.preprocessor.profile_extractor") as mock_profile,
        patch("src.agents.intent_classifier.intent_classifier") as mock_intent,
        patch("src.shared.guardrails.guardrail_manager") as mock_guard,
    ):
        # Use AsyncMock for async methods
        mock_rewriter.rewrite = AsyncMock(return_value="query")
        mock_intent.classify = AsyncMock(
            return_value=Intent.COMPLEX_PROCEDURE
        )  # To trigger Slow Lane
        mock_guard.validate_topic = AsyncMock(return_value=(True, ""))
        mock_profile.extract = AsyncMock(return_value={})

        orchestrator = AdminOrchestrator()

        # Run stream_query
        events = []
        async for event in orchestrator.stream_query("query", "fr", "test"):
            if event["type"] == "token":
                events.append(event["content"])

        print(f"Captured events: {events}")

        assert "Thinking..." not in events, "Failed to filter 'internal' event"
        assert "Hello user." in events, "Failed to yield standard event"
        print("SUCCESS: Stream filtering works.")


if __name__ == "__main__":
    asyncio.run(test_stream_filtering())

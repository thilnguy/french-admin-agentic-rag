import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import AgentState
from src.agents.intent_classifier import Intent
from src.shared.query_pipeline import PipelineResult


async def test_stream_filtering():
    print("Testing Stream Filtering...")

    # Mock dependencies
    mock_memory = AsyncMock()
    mock_memory.load_agent_state.return_value = AgentState(session_id="test")

    # Mock Graph
    mock_graph = AsyncMock()

    # Simulate a stream of events from the graph
    async def mock_astream_events(*args, **kwargs):
        # Helper to create a chunk with content
        def create_chunk(text):
            m = MagicMock()
            m.content = text
            return m

        events = [
            # Event 1: Internal thinking (Should be filtered)
            {
                "event": "on_chat_model_stream",
                "tags": ["internal"],
                "data": {"chunk": create_chunk("Thinking...")},
            },
            # Event 2: Real answer (Should be yielded)
            {
                "event": "on_chat_model_stream",
                "tags": [],
                "data": {"chunk": create_chunk("Hello user.")},
            },
        ]
        for e in events:
            yield e

    mock_graph.astream_events.side_effect = mock_astream_events

    # Create Orchestrator
    with (
        patch("src.agents.orchestrator.memory_manager", mock_memory),
        patch("src.agents.orchestrator.agent_graph") as mock_agent_graph,
        patch("src.agents.preprocessor.query_rewriter") as mock_rewriter,
        patch("src.agents.preprocessor.profile_extractor") as mock_profile,
        patch("src.agents.intent_classifier.intent_classifier") as mock_intent,
        patch("src.shared.guardrails.guardrail_manager") as mock_guard,
        patch("src.agents.orchestrator.redis.Redis") as mock_redis,
        patch("src.agents.orchestrator.ChatOpenAI"),
        patch(
            "src.agents.orchestrator.translate_admin_text", new_callable=AsyncMock
        ) as mock_translator,
        patch(
            "src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retriever,
        patch("src.shared.query_pipeline.get_query_pipeline") as mock_get_pipeline,
    ):
        mock_redis.return_value = AsyncMock()
        mock_redis.return_value.get.return_value = None
        mock_agent_graph.astream_events.side_effect = mock_astream_events
        # Setup Translator/Retriever
        mock_translator.return_value = "Mocked translation"
        mock_retriever.return_value = []
        # Setup Pipeline Mock
        mock_pipeline = AsyncMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.run.return_value = PipelineResult(
            rewritten_query="query",
            intent=Intent.COMPLEX_PROCEDURE,
            extracted_data={},
            new_core_goal=None,
        )

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

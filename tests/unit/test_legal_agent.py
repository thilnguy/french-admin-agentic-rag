import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.legal_agent import LegalResearchAgent
from src.agents.state import AgentState


@pytest.mark.asyncio
async def test_legal_agent_run_flow():
    # Mock dependencies
    with patch("src.agents.legal_agent.ChatOpenAI") as mock_llm_cls, patch(
        "src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock
    ) as mock_retrieve:
        # Setup LLM Mock
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # We need to mock the chains or the llm.invoke
        # Since implementation uses `self.refiner_prompt | self.llm | StrOutputParser()`,
        # it's harder to mock the chain components individually without refactoring.
        # But we can mock the `ainvoke` of the chain if we mock `ChatPromptTemplate`...
        # Actually easier to mock the `_refine_query` etc methods if I want to test the flow,
        # but I want to test the `run` method which calls them.

        # Let's mock the internal helper methods of the instance for white-box testing of `run`
        # This avoids complex chain mocking.

        agent = LegalResearchAgent()

        # Check that __init__ set up prompts
        assert agent.refiner_prompt is not None

        # Mock internal methods
        agent._refine_query = AsyncMock(return_value="refined query")
        agent._evaluate_context = AsyncMock(return_value=True)  # Sufficient
        agent._synthesize_answer = AsyncMock(return_value="Final Answer")
        agent._format_docs = MagicMock(return_value="formatted docs")

        mock_retrieve.return_value = [
            {"source": "s1", "content": "c1", "metadata": {"title": "t1"}}
        ]

        # Run
        state = AgentState(session_id="test", messages=[])
        response = await agent.run("original query", state)

        # Verify Flow
        agent._refine_query.assert_called_with("original query")
        mock_retrieve.assert_called_with("refined query", domain="general")
        agent._evaluate_context.assert_called_with("original query", "formatted docs")
        agent._synthesize_answer.assert_called_with("original query", "formatted docs")

        assert response == "Final Answer"


@pytest.mark.asyncio
async def test_legal_agent_insufficient_context():
    # Test handling of empty or insufficient docs
    with patch("src.agents.legal_agent.ChatOpenAI"):
        agent = LegalResearchAgent()

        # Mock methods
        agent._refine_query = AsyncMock(return_value="refined")
        agent._evaluate_context = AsyncMock(return_value=False)  # Insufficient
        agent._synthesize_answer = AsyncMock(return_value="Fallback Answer")
        agent._format_docs = MagicMock(return_value="docs")

        with patch(
            "src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retrieve:
            mock_retrieve.return_value = [{"content": "weak info"}]

            response = await agent.run("query", None)

            # Should still synthesize (fallback logic in code)
            agent._synthesize_answer.assert_called()
            assert response == "Fallback Answer"

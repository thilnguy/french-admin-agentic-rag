import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.legal_agent import LegalResearchAgent
from src.agents.state import AgentState


@pytest.mark.asyncio
async def test_legal_agent_run_flow():
    # Mock dependencies
    with (
        patch("src.agents.legal_agent.ChatOpenAI") as mock_llm_cls,
        patch("src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock),
    ):
        # Setup LLM Mock
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # We need to mock the chains or the llm.invoke


# The original test_legal_agent_run_flow will be replaced by the new tests.


@pytest.mark.asyncio
async def test_refine_query_logic():
    with patch("src.agents.legal_agent.ChatOpenAI"):
        agent = LegalResearchAgent()
        agent._run_chain = AsyncMock(return_value="Refined Query")

        refined = await agent._refine_query("raw query")
        assert refined == "Refined Query"
        agent._run_chain.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_answer_logic():
    with patch("src.agents.legal_agent.ChatOpenAI"):
        agent = LegalResearchAgent()
        agent._run_chain = AsyncMock(return_value="Final Answer")

        # Case 1: Context present
        # Updated signature: query, context, user_lang
        ans = await agent._synthesize_answer("query", "context", "fr")
        assert ans == "Final Answer"

        # Case 2: No context
        ans_empty = await agent._synthesize_answer("query", "", "fr")
        assert "Je n'ai trouvé aucune information" in ans_empty


@pytest.mark.asyncio
async def test_legal_agent_run_full_flow():
    with (
        patch("src.agents.legal_agent.ChatOpenAI"),
        patch(
            "src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retrieve,
    ):
        agent = LegalResearchAgent()

        agent._refine_query = AsyncMock(return_value="refined")
        # agent._evaluate_context = AsyncMock(return_value=True) # Removed
        agent._synthesize_answer = AsyncMock(return_value="answer")

        mock_retrieve.return_value = [
            {"content": "doc", "source": "url", "metadata": {"title": "T"}}
        ]

        from src.agents.state import UserProfile

        state = AgentState(
            session_id="test", messages=[], user_profile=UserProfile(language="fr")
        )
        res = await agent.run("query", state)

        assert res == "answer"
        agent._refine_query.assert_called_with("query")
        mock_retrieve.assert_called()
        agent._synthesize_answer.assert_called()


@pytest.mark.asyncio
async def test_legal_agent_insufficient_context():
    # Test handling of empty or insufficient docs
    with patch("src.agents.legal_agent.ChatOpenAI"):
        agent = LegalResearchAgent()

        # Mock methods
        agent._refine_query = AsyncMock(return_value="refined")
        # agent._evaluate_context removed
        agent._synthesize_answer = AsyncMock(return_value="Fallback Answer")
        agent._format_docs = MagicMock(return_value="docs")

        with patch(
            "src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retrieve:
            mock_retrieve.return_value = [{"content": "weak info"}]

            from src.agents.state import UserProfile

            state = AgentState(
                session_id="test", messages=[], user_profile=UserProfile(language="fr")
            )
            response = await agent.run("query", state)

            # Should still synthesize (fallback logic in code)
            agent._synthesize_answer.assert_called()
            assert response == "Fallback Answer"

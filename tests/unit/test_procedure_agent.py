import pytest
from unittest.mock import AsyncMock, patch
from src.agents.procedure_agent import ProcedureGuideAgent
from src.agents.state import AgentState, UserProfile


@pytest.mark.asyncio
async def test_determine_step_logic():
    with patch("src.agents.procedure_agent.ChatOpenAI"):
        agent = ProcedureGuideAgent()
        # Mock _run_chain to return specific steps
        agent._run_chain = AsyncMock(return_value="CLARIFICATION")

        step = await agent._determine_step("query", {}, "history")
        assert step == "CLARIFICATION"
        agent._run_chain.assert_called_once()


@pytest.mark.asyncio
async def test_ask_clarification_logic():
    with patch("src.agents.procedure_agent.ChatOpenAI"):
        agent = ProcedureGuideAgent()
        agent._run_chain = AsyncMock(return_value="Quelle est votre nationalité ?")

        state = AgentState(session_id="test", messages=[], user_profile=UserProfile())
        question = await agent._ask_clarification("query", state)
        assert question == "Quelle est votre nationalité ?"


@pytest.mark.asyncio
async def test_explain_procedure_logic():
    with patch("src.agents.procedure_agent.ChatOpenAI"):
        agent = ProcedureGuideAgent()
        agent._run_chain = AsyncMock(return_value="Step 1: Do this.")

        # Case 1: Docs provided
        response = await agent._explain_procedure("query", [{"content": "doc1"}])
        assert response == "Step 1: Do this."

        # Case 2: No docs
        response_empty = await agent._explain_procedure("query", [])
        assert "Je ne trouve pas de procédure" in response_empty


@pytest.mark.asyncio
async def test_full_run_integration_mocked_llm():
    """Test the run() method logic linking the steps."""
    with (
        patch("src.agents.procedure_agent.ChatOpenAI"),
        patch(
            "src.agents.procedure_agent.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retrieve,
    ):
        agent = ProcedureGuideAgent()

        # We need to mock _determine_step to control flow, OR mock _run_chain to return "RETRIEVAL" first
        # Let's mock _determine_step for simplicity in this flow test,
        # as we tested _determine_step logic separately above.
        # BUT to increase coverage we should let it run if possible.
        # However, _run_chain is generic.

        # Strategy: Mock _run_chain to handle different prompts?
        # Too complex. Let's stick to mocking helper methods for the run() test
        # AND keeping the unit tests above for the helper methods.

        agent._determine_step = AsyncMock(return_value="RETRIEVAL")
        agent._ask_clarification = AsyncMock(return_value="Clarify?")
        agent._explain_procedure = AsyncMock(return_value="Done.")

        mock_retrieve.return_value = [{"content": "doc"}]

        state = AgentState(session_id="test", messages=[], user_profile=UserProfile())
        res = await agent.run("how to", state)

        assert res == "Done."
        assert state.current_step == "RETRIEVAL"


@pytest.mark.asyncio
async def test_procedure_agent_fallback_and_explanation():
    with (
        patch("src.agents.procedure_agent.ChatOpenAI"),
        patch(
            "src.agents.procedure_agent.retrieve_legal_info", new_callable=AsyncMock
        ) as mock_retrieve,
    ):
        agent = ProcedureGuideAgent()

        # Test 1: Fallback (Unknown step)
        agent._determine_step = AsyncMock(return_value="UNKNOWN_STEP")
        agent._explain_procedure = AsyncMock(return_value="Fallback Guide")
        mock_retrieve.return_value = [{"content": "content"}]

        state = AgentState(session_id="test", messages=[])
        res = await agent.run("query", state)

        assert res == "Fallback Guide"
        mock_retrieve.assert_called()

        # Test 2: EXPLANATION step (Pass through to fallback currently)
        agent._determine_step = AsyncMock(return_value="EXPLANATION")
        res2 = await agent.run("query", state)
        assert res2 == "Fallback Guide"

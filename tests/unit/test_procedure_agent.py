import pytest
from unittest.mock import AsyncMock, patch
from src.agents.procedure_agent import ProcedureGuideAgent
from src.agents.state import AgentState, UserProfile


@pytest.mark.asyncio
async def test_procedure_agent_clarification_flow():
    with patch("src.agents.procedure_agent.ChatOpenAI"):
        agent = ProcedureGuideAgent()

        # Mock internal methods
        agent._determine_step = AsyncMock(return_value="CLARIFICATION")
        agent._ask_clarification = AsyncMock(return_value="Asking nicely")
        agent._explain_procedure = AsyncMock(return_value="Procedure Explained")

        # State
        state = AgentState(session_id="test", messages=[], user_profile=UserProfile())

        # Run
        response = await agent.run("procedure query", state)

        # Verify
        agent._determine_step.assert_called_with(
            "procedure query", state.user_profile.model_dump(), ""
        )
        agent._ask_clarification.assert_called()
        agent._explain_procedure.assert_not_called()
        assert state.current_step == "CLARIFICATION"
        assert response == "Asking nicely"


@pytest.mark.asyncio
async def test_procedure_agent_retrieval_flow():
    with patch("src.agents.procedure_agent.ChatOpenAI"), patch(
        "src.agents.procedure_agent.retrieve_legal_info", new_callable=AsyncMock
    ) as mock_retrieve:
        agent = ProcedureGuideAgent()

        agent._determine_step = AsyncMock(return_value="RETRIEVAL")
        agent._ask_clarification = AsyncMock()
        agent._explain_procedure = AsyncMock(return_value="Final Guide")

        mock_retrieve.return_value = [{"content": "proc content"}]

        state = AgentState(session_id="test", messages=[], user_profile=UserProfile())

        response = await agent.run("how to passport", state)

        agent._determine_step.assert_called()
        mock_retrieve.assert_called_with("how to passport", domain="procedure")
        agent._explain_procedure.assert_called()
        assert (
            state.current_step == "RETRIEVAL"
        )  # Or updated to something else if logic changes
        assert response == "Final Guide"

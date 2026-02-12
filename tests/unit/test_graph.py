import pytest
from unittest.mock import AsyncMock, patch
from src.agents.graph import (
    legal_expert_node,
    procedure_expert_node,
    route_request,
    END,
)
from src.agents.state import AgentState
from src.agents.intent_classifier import Intent
from langchain_core.messages import HumanMessage


@pytest.mark.asyncio
async def test_legal_expert_node():
    with patch("src.agents.graph.legal_agent") as mock_agent:
        mock_agent.run = AsyncMock(return_value="Legal Answer")

        state = AgentState(
            session_id="test",
            messages=[HumanMessage(content="Question")],
            intent=Intent.LEGAL_INQUIRY,
        )

        result = await legal_expert_node(state)

        mock_agent.run.assert_called_once()
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Legal Answer"


@pytest.mark.asyncio
async def test_procedure_expert_node():
    with patch("src.agents.graph.procedure_agent") as mock_agent:
        mock_agent.run = AsyncMock(return_value="Procedure Step")

        state = AgentState(
            session_id="test",
            messages=[HumanMessage(content="How to?")],
            intent=Intent.COMPLEX_PROCEDURE,
        )

        result = await procedure_expert_node(state)

        mock_agent.run.assert_called_once()
        assert result["messages"][0].content == "Procedure Step"


def test_route_request():
    # Test Legal
    state_legal = AgentState(session_id="1", messages=[], intent=Intent.LEGAL_INQUIRY)
    assert route_request(state_legal) == "legal_expert"

    # Test Procedure
    state_proc = AgentState(
        session_id="2", messages=[], intent=Intent.COMPLEX_PROCEDURE
    )
    assert route_request(state_proc) == "procedure_expert"

    state_form = AgentState(session_id="3", messages=[], intent=Intent.FORM_FILLING)
    assert route_request(state_form) == "procedure_expert"

    # Test End / Default
    state_simple = AgentState(session_id="4", messages=[], intent=Intent.SIMPLE_QA)
    assert route_request(state_simple) == END

    state_unknown = AgentState(session_id="5", messages=[], intent=Intent.UNKNOWN)
    assert route_request(state_unknown) == END

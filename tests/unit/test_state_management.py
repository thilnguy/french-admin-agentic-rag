import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.memory.manager import MemoryManager
from src.agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def mock_memory_manager():
    with patch("src.memory.manager.redis.from_url") as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        mgr = MemoryManager()
        # Explicitly replace the client with strict mock
        mgr.redis_client = mock_client
        return mgr


@pytest.mark.asyncio
async def test_save_and_load_state(mock_memory_manager):
    session_id = "test_session"
    state = AgentState(
        session_id=session_id,
        messages=[HumanMessage(content="Hello"), AIMessage(content="Hi")],
        intent="SIMPLE_QA",
    )

    # Test Save
    await mock_memory_manager.save_agent_state(session_id, state)

    # Verify Redis set call
    mock_memory_manager.redis_client.set.assert_called_once()
    args, _ = mock_memory_manager.redis_client.set.call_args
    key, val = args
    assert key == f"agent_state:{session_id}"

    saved_data = json.loads(val)
    assert saved_data["intent"] == "SIMPLE_QA"
    assert len(saved_data["messages"]) == 2
    assert saved_data["messages"][0]["type"] == "human"

    # Test Load
    mock_memory_manager.redis_client.get.return_value = val
    loaded_state = await mock_memory_manager.load_agent_state(session_id)

    assert loaded_state.session_id == session_id
    assert len(loaded_state.messages) == 2
    assert isinstance(loaded_state.messages[0], HumanMessage)
    assert loaded_state.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_migration_from_legacy(mock_memory_manager):
    session_id = "legacy_session"

    # 1. Simulate NO existing state (cache miss on new key)
    mock_memory_manager.redis_client.get.return_value = None

    # 2. Simulate YES existing legacy messages
    with patch.object(mock_memory_manager, "get_session_history") as mock_get_history:
        mock_history = MagicMock()
        mock_history.messages = [HumanMessage(content="Legacy Msg")]
        mock_get_history.return_value = mock_history

        # Test Load
        state = await mock_memory_manager.load_agent_state(session_id)

        # Should have converted legacy messages
        assert len(state.messages) == 1
        assert state.messages[0].content == "Legacy Msg"

        # Should have saved new state immediately
        mock_memory_manager.redis_client.set.assert_called_once()

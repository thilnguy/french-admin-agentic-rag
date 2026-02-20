import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator
from unittest.mock import MagicMock, AsyncMock, patch
from src.main import app
import os

# Mock environment variables BEFORE any other imports could trigger logic
os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
os.environ["QDRANT_API_KEY"] = "mock-qdrant-key"
os.environ["TEST_MODE"] = "True"

# Force using a specific event loop scope if needed,
# but generic pytest-asyncio auto handling is usually sufficient in newer versions.


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_openai_globally():
    """Automatically mock ChatOpenAI for all tests to prevent unauthenticated network calls."""
    from langchain_core.messages import AIMessage

    def create_mock_msg(*args, **kwargs):
        # Return a JSON string that also contains "APPROVED" to satisfy both
        # JsonOutputParser (ProfileExtractor) and topic validation (GuardrailManager).
        return AIMessage(
            content='{"language": "fr", "status": "APPROVED", "answer": "Mocked LLM response"}'
        )

    # Patch the methods on the class so that already-instantiated singletons are affected
    with (
        patch(
            "langchain_openai.ChatOpenAI.ainvoke",
            new_callable=AsyncMock,
            side_effect=lambda *args, **kwargs: create_mock_msg(),
        ),
        patch(
            "langchain_openai.ChatOpenAI.invoke", MagicMock(side_effect=create_mock_msg)
        ),
        patch(
            "langchain_openai.ChatOpenAI.astream", new_callable=AsyncMock
        ) as mock_astream,
        patch(
            "langchain_openai.ChatOpenAI.stream",
            MagicMock(return_value=[create_mock_msg()]),
        ),
    ):

        async def mock_astream_gen(*args, **kwargs):
            yield AIMessage(content="Mocked ")
            yield AIMessage(content="LLM ")
            yield AIMessage(content="response")

        mock_astream.side_effect = mock_astream_gen

        yield


@pytest_asyncio.fixture
async def ac() -> AsyncGenerator[AsyncClient, None]:
    """
    Async client fixture for FastAPI app.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

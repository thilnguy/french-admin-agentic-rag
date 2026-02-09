import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from typing import AsyncGenerator
from src.main import app
from src.config import settings
import os

# Mock environment variables BEFORE any other imports could trigger logic
os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
os.environ["QDRANT_API_KEY"] = "mock-qdrant-key"

# Force using a specific event loop scope if needed, 
# but generic pytest-asyncio auto handling is usually sufficient in newer versions.

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def ac() -> AsyncGenerator[AsyncClient, None]:
    """
    Async client fixture for FastAPI app.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

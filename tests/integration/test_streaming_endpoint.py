import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from src.main import app
import json


@pytest.mark.asyncio
async def test_chat_stream_endpoint():
    # Mock orchestrator.stream_query
    async def mock_stream_query(query, user_lang, session_id):
        yield {"type": "status", "content": "Thinking..."}
        yield {"type": "token", "content": "Hello"}
        yield {"type": "token", "content": " World"}

    with patch("src.main.orchestrator.stream_query", side_effect=mock_stream_query):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "GET",
                "/chat/stream",
                params={"query": "Test", "language": "fr", "session_id": "123"},
                headers={"X-API-Key": "test-key"},
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                events = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        events.append(json.loads(data))

                assert len(events) == 3
                assert events[0] == {"type": "status", "content": "Thinking..."}
                assert events[1] == {"type": "token", "content": "Hello"}
                assert events[2] == {"type": "token", "content": " World"}


# ---------------------------------------------------------------------------
# Fix 1: Input Validation Tests (added after adding Query validators to /chat/stream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_empty_query_returns_422():
    """Empty query string should be rejected with 422 Unprocessable Entity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/chat/stream",
            params={"query": "", "language": "fr"},
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_stream_oversized_query_returns_422():
    """Query exceeding 500 characters should be rejected with 422."""
    oversized = "A" * 501
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/chat/stream",
            params={"query": oversized, "language": "fr"},
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_stream_invalid_language_returns_422():
    """Language not in (fr, en, vi) should be rejected with 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/chat/stream",
            params={"query": "Hello", "language": "zh"},
            headers={"X-API-Key": "test-key"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_stream_valid_request_passes_validation():
    """Valid request with all constraints satisfied reaches the orchestrator."""

    async def mock_stream_query(query, user_lang, session_id):
        yield {"type": "token", "content": "OK"}

    with patch("src.main.orchestrator.stream_query", side_effect=mock_stream_query):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "GET",
                "/chat/stream",
                params={"query": "Bonjour", "language": "en", "session_id": "sess1"},
                headers={"X-API-Key": "test-key"},
            ) as response:
                assert response.status_code == 200

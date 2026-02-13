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
                "POST",
                "/chat/stream",
                json={"query": "Test", "language": "fr", "session_id": "123"},
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

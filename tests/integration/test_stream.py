import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_chat_stream_endpoint(ac: AsyncClient):
    """Chat stream endpoint should yield SSE JSON events when orchestrator is mocked."""
    
    async def mock_stream_query(query, language, session_id):
        yield {"type": "status", "content": "Analysing request..."}
        yield {"type": "token", "content": "Test"}
        yield {"type": "token", "content": " response"}

    with patch("src.main.orchestrator") as mock_orch:
        mock_orch.stream_query = mock_stream_query

        # Streaming responses in httpx require reading the stream or just fetching it.
        # Since FastAPI StreamingResponse sends data as it comes, we can just GET request.
        # Ensure httpx gets the whole response text or we can iterate over lines.
        response = await ac.get(
            "/chat/stream",
            params={"query": "Comment faire un passeport ?", "language": "fr"}
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        text = response.text
        # Check standard SSE formatting
        assert "data: {\"type\": \"status\", \"content\": \"Analysing request...\"}\n\n" in text
        assert "data: {\"type\": \"token\", \"content\": \"Test\"}\n\n" in text
        assert "data: {\"type\": \"token\", \"content\": \" response\"}\n\n" in text
        assert "data: [DONE]\n\n" in text

@pytest.mark.asyncio
async def test_chat_stream_validation(ac: AsyncClient):
    """Missing or invalid fields in stream should return 422."""
    # Missing query
    response = await ac.get("/chat/stream", params={"language": "fr"})
    assert response.status_code == 422

    # Invalid language
    response = await ac.get("/chat/stream", params={"query": "hello", "language": "xx"})
    assert response.status_code == 422

    # Query too long
    long_query = "a" * 501
    response = await ac.get("/chat/stream", params={"query": long_query, "language": "fr"})
    assert response.status_code == 422

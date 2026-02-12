import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(ac: AsyncClient):
    """Health endpoint should return 200 with status and dependencies."""
    with patch("redis.from_url") as mock_redis_from_url, patch(
        "qdrant_client.QdrantClient"
    ) as mock_qdrant_cls:
        # Mock Redis ping
        mock_r = mock_redis_from_url.return_value
        mock_r.ping.return_value = True

        # Mock Qdrant get_collections
        mock_q = mock_qdrant_cls.return_value
        mock_q.get_collections.return_value = []

        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "dependencies" in data


@pytest.mark.asyncio
async def test_chat_endpoint_validation(ac: AsyncClient):
    """Missing or invalid fields should return 422."""
    # Missing query
    response = await ac.post("/chat", json={"language": "fr"})
    assert response.status_code == 422

    # Invalid language
    response = await ac.post("/chat", json={"query": "hello", "language": "xx"})
    assert response.status_code == 422

    # Query too long
    long_query = "a" * 501
    response = await ac.post("/chat", json={"query": long_query, "language": "fr"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_api_key_enforcement(ac: AsyncClient):
    """API Key should be enforced if configured."""
    # Patch settings.API_KEY to enable auth
    with patch("src.main.settings.API_KEY", "test-secret"):
        # 1. Missing Key -> 403
        response = await ac.post("/chat", json={"query": "hello", "language": "fr"})
        assert response.status_code == 403

        # 2. Wrong Key -> 403
        response = await ac.post(
            "/chat",
            json={"query": "hello", "language": "fr"},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 403

        # 3. Correct Key -> 200 (Mocked Orchestrator)
        with patch("src.main.orchestrator") as mock_orch:
            mock_orch.handle_query = AsyncMock(return_value="Answer")
            response = await ac.post(
                "/chat",
                json={"query": "hello", "language": "fr"},
                headers={"X-API-Key": "test-secret"},
            )
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_flow_mocked(ac: AsyncClient):
    """Chat endpoint should return 200 with answer when orchestrator is mocked."""
    with patch("src.main.orchestrator") as mock_orch:
        mock_orch.handle_query = AsyncMock(return_value="Mocked answer for testing")

        response = await ac.post(
            "/chat", json={"query": "Comment faire un passeport ?", "language": "fr"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Mocked answer for testing"
        mock_orch.handle_query.assert_called_once()


@pytest.mark.asyncio
async def test_root_endpoint(ac: AsyncClient):
    """Root endpoint should return online status."""
    response = await ac.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "French Admin Agent" in data["status"]

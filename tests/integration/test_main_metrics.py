import pytest
from httpx import AsyncClient
from src.utils import metrics


@pytest.mark.asyncio
async def test_feedback_endpoint_metrics(ac: AsyncClient):
    """Test that feedback endpoint increments the metric."""

    # Initial count
    try:
        initial_pos = metrics.USER_FEEDBACK.labels(score="positive")._value.get()
    except Exception:
        initial_pos = 0

    response = await ac.post(
        "/feedback", json={"session_id": "123", "score": "positive"}
    )
    assert response.status_code == 200

    # Check metric increment
    new_pos = metrics.USER_FEEDBACK.labels(score="positive")._value.get()
    assert new_pos == initial_pos + 1


@pytest.mark.asyncio
async def test_metrics_endpoint_exposed(ac: AsyncClient):
    """Test that /metrics endpoint is exposed by instrumentator."""
    response = await ac.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text

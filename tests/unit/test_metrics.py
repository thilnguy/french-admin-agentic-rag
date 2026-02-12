from src.utils import metrics


def test_metrics_initialization():
    assert metrics.LLM_TOKEN_USAGE is not None
    assert metrics.LLM_REQUEST_DURATION is not None
    assert metrics.RAG_RETRIEVAL_LATENCY is not None
    assert metrics.RAG_GENERATION_LATENCY is not None
    assert metrics.USER_FEEDBACK is not None


def test_metrics_labels():
    # Attempt to increment/observe to ensure labels match definition
    metrics.LLM_TOKEN_USAGE.labels(model="gpt-4o", type="prompt").inc(10)
    metrics.LLM_REQUEST_DURATION.labels(model="gpt-4o").observe(0.5)
    metrics.RAG_RETRIEVAL_LATENCY.labels(domain="general").observe(0.2)
    metrics.USER_FEEDBACK.labels(score="positive").inc()

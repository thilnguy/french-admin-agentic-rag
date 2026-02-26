from prometheus_client import Counter, Histogram

# LLM Metrics
LLM_TOKEN_USAGE = Counter(
    "llm_token_usage_total",
    "Total number of tokens used by LLM",
    ["model", "type"],  # type: prompt, completion
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "Time spent waiting for LLM response",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# RAG Metrics
RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving documents from vector DB",
    ["domain"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

RAG_GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Time spent generating final answer",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0],
)

RERANKER_LATENCY = Histogram(
    "reranker_latency_seconds",
    "Time spent in cross-encoder reranking layer",
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0],
)

# Business Metrics
USER_FEEDBACK = Counter(
    "user_feedback_total",
    "User feedback scores",
    ["score"],  # positive, negative
)

GUARDRAIL_REJECTIONS = Counter(
    "guardrail_rejections_total",
    "Total number of queries rejected by guardrails",
    ["reason"],
)

TOPIC_DETECTION = Counter(
    "topic_detection_total",
    "Total queries classified by topic",
    ["topic"],
)

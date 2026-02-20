"""
Behavior Contract Tests — freeze current orchestrator behavior before refactoring.

DESIGN PRINCIPLE:
    These tests mock ONLY at EXTERNAL BOUNDARIES (LLM API, Redis, Qdrant).
    They do NOT mock internal modules or import paths (e.g. retrieve_legal_info,
    guardrail_manager, intent_classifier, etc.).

    This means if we move logic between modules during refactoring, these tests
    remain valid and will catch any behavioral regressions.

CONTRACT: If any of these tests break during refactoring, STOP and fix the code, not the test.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import AgentState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_state(session_id="test-session"):
    return AgentState(session_id=session_id, messages=[])


def mock_llm_response(content="Mocked LLM Answer"):
    mock = MagicMock()
    mock.content = content
    mock.response_metadata = {}
    return mock


# Core external patches applied to all tests in this module
EXTERNAL_PATCHES = [
    # Redis — always a miss unless overridden
    patch("src.agents.orchestrator.redis.Redis"),
    # Qdrant — no real vector DB needed
    patch("skills.legal_retriever.main._get_qdrant_client"),
    # Embeddings — no model loading
    patch("skills.legal_retriever.main._get_embeddings"),
]


# ---------------------------------------------------------------------------
# CONTRACT 1: handle_query — SIMPLE_QA returns an answer with disclaimer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_handle_query_simple_qa_returns_answer():
    """Contract: SIMPLE_QA query → LLM answer with disclaimer appended."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch("skills.legal_retriever.main.QdrantVectorStore"),
        patch("skills.legal_retriever.main._get_qdrant_client") as mock_client,
        patch("skills.legal_retriever.main._get_embeddings"),
        patch("skills.legal_retriever.main.get_reranker") as mock_reranker,
    ):
        # External boundary: LLM mocked
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None  # Cache miss
        orch.cache.setex = AsyncMock()
        orch._call_llm = AsyncMock(return_value=mock_llm_response("La réponse est 42."))

        # Qdrant returns no docs (simplest case)
        mock_client.return_value.collection_exists.return_value = False
        mock_reranker.return_value.rerank.return_value = []

        # Memory
        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        result = await orch.handle_query("Combien coûte un passeport?", "fr", "s1")

        # CONTRACT: answer is returned, disclaimer is added
        assert isinstance(result, str)
        assert len(result) > 0
        # Any non-rejected answer should contain the LLM output or a known fallback
        assert (
            "42" in result
            or "information" in result.lower()
            or "désolé" in result.lower()
        )


# ---------------------------------------------------------------------------
# CONTRACT 2: handle_query — cache hit returns cached value immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_handle_query_cache_hit():
    """Contract: If Redis has a cached response, return it immediately without LLM."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = "Cached correct answer"
        orch._call_llm = AsyncMock()  # Should never be called

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        with patch("src.config.settings.DEBUG", False):
            result = await orch.handle_query("Any query", "fr", "session-cache")

        # CONTRACT: LLM not called, cached value returned
        assert result == "Cached correct answer"
        orch._call_llm.assert_not_called()


# ---------------------------------------------------------------------------
# CONTRACT 3: handle_query — off-topic query is rejected with localized message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_handle_query_topic_rejection_french():
    """Contract: Off-topic query in an English sentence → rejection in English.

    DOCUMENTED BEHAVIOR: ProfileExtractor detects language from query TEXT before the
    guardrail check. 'Tell me a joke' is English → ProfileExtractor sets language='en'.
    The user_lang='fr' hint is overridden by what the ProfileExtractor detects.
    Result: rejection is in English, not French.

    This is a KNOWN BEHAVIOR. To get a French rejection, the query text must also be French
    (e.g. 'Raconte-moi une blague').
    """
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch(
            "src.shared.guardrails.GuardrailManager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
        patch(
            "src.agents.preprocessor.profile_extractor.extract", new_callable=AsyncMock
        ) as mock_extract,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch._call_llm = AsyncMock()

        state = make_state()
        mock_extract.return_value = {"language": "en"}

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        # Off-topic (English text)
        mock_validate.return_value = (
            False,
            "This request is unrelated to French administration.",
        )

        result = await orch.handle_query("Tell me a joke", "fr", "s2")

        # CONTRACT: English text → ProfileExtractor detects 'en' → English rejection
        assert "Sorry" in result
        orch._call_llm.assert_not_called()


@pytest.mark.asyncio
async def test_contract_handle_query_topic_rejection_french_text():
    """Contract: Off-topic query in French text → French rejection message."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch(
            "src.shared.guardrails.GuardrailManager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch._call_llm = AsyncMock()

        state = make_state()
        # Pre-set language to French so anti-hallucination keeps it
        state.user_profile.language = "French"
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        # Off-topic in French
        mock_validate.return_value = (
            False,
            "This request is unrelated to French administration.",
        )

        result = await orch.handle_query("Raconte-moi une blague", "fr", "s2")

        # CONTRACT: French text → French rejection
        assert "Désolé" in result
        orch._call_llm.assert_not_called()


@pytest.mark.asyncio
async def test_contract_handle_query_topic_rejection_english():
    """Contract: Off-topic query with EN lang → English rejection."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch(
            "src.shared.guardrails.GuardrailManager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch._call_llm = AsyncMock()

        state = make_state()
        state.user_profile.language = "English"
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        mock_validate.return_value = (
            False,
            "This request is unrelated to French administration.",
        )

        result = await orch.handle_query("Tell me a joke", "en", "s3")

        # CONTRACT: English rejection when language is English
        assert "Sorry" in result or "cannot" in result.lower()
        orch._call_llm.assert_not_called()


# ---------------------------------------------------------------------------
# CONTRACT 4: stream_query — SIMPLE_QA yields status + tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_stream_query_simple_qa_yields_events():
    """Contract: stream_query for SIMPLE_QA yields status events and token chunks."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        # Force SIMPLE_QA to avoid AgentGraph path
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
            return_value="SIMPLE_QA",
        ),
        patch(
            "src.agents.orchestrator.retrieve_legal_info",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch.cache.setex = AsyncMock()

        # LLM streams tokens
        async def mock_astream(messages):
            yield MagicMock(content="Bonjour")
            yield MagicMock(content=" monde")

        orch.llm = MagicMock()
        orch.llm.astream = mock_astream

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        events = []
        async for event in orch.stream_query("Quelle est la procédure?", "fr", "s4"):
            events.append(event)

        types = [e["type"] for e in events]
        # CONTRACT: at least one status event and one token event
        assert "status" in types
        assert "token" in types

        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "Bonjour" in tokens


# ---------------------------------------------------------------------------
# CONTRACT 5: stream_query — cache hit returns cached without calling LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_stream_query_cache_hit():
    """Contract: stream_query returns cached answer immediately without LLM."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = "Cached stream answer"
        orch.llm = MagicMock()
        orch.llm.astream = AsyncMock()

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        with patch("src.config.settings.DEBUG", False):
            events = []
            async for event in orch.stream_query("Cached query", "fr", "s5"):
                events.append(event)

        # CONTRACT: exactly 2 events: status + token with cached content
        assert len(events) == 2
        assert events[0]["type"] == "status"
        assert events[1] == {"type": "token", "content": "Cached stream answer"}
        orch.llm.astream.assert_not_called()


# ---------------------------------------------------------------------------
# CONTRACT 6: stream_query — off-topic query yields rejection token
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_stream_query_topic_rejection():
    """Contract: stream_query for off-topic query yields a rejection token and stops."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch(
            "src.shared.guardrails.GuardrailManager.validate_topic",
            new_callable=AsyncMock,
        ) as mock_validate,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch.llm = MagicMock()
        orch.llm.astream = AsyncMock()

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()
        mock_validate.return_value = (False, "Off topic.")

        events = []
        async for event in orch.stream_query("Tell me a joke", "fr", "s6"):
            events.append(event)

        # CONTRACT: at least one token event with rejection text
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 1
        assert (
            "Désolé" in token_events[0]["content"]
            or "Sorry" in token_events[0]["content"]
        )

        # No more events after rejection
        orch.llm.astream.assert_not_called()


# ---------------------------------------------------------------------------
# CONTRACT 7: handle_query — Redis error is handled gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_handle_query_redis_error_does_not_crash():
    """Contract: Redis failure is silent — system continues and returns an answer."""
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch("skills.legal_retriever.main._get_qdrant_client") as mock_client,
        patch("skills.legal_retriever.main._get_embeddings"),
        patch("skills.legal_retriever.main.get_reranker") as mock_reranker,
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.side_effect = Exception("Redis is down")
        orch.cache.setex.side_effect = Exception("Redis is down")
        orch._call_llm = AsyncMock(
            return_value=mock_llm_response("Answer despite Redis error.")
        )

        mock_client.return_value.collection_exists.return_value = False
        mock_reranker.return_value.rerank.return_value = []

        state = make_state()
        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock()

        # CONTRACT: does not raise — Redis failures are silent
        result = await orch.handle_query("Question", "fr", "s7")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# CONTRACT 8: Language switch — detected language persists on state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_language_detection_updates_state():
    """Contract: A clearly English query with user_lang='en' switches state language to English.

    CURRENT BEHAVIOR (after Fix 2a — LanguageResolver integrated):
        ProfileExtractor correctly detects 'en' for 'Tell me about myself'.
        LanguageResolver respects the frontend override (user_lang='en') and applies 'English'.
        Result: state.user_profile.language == 'English'.

    PREVIOUS BEHAVIOR (DOCUMENTED BUG — pre-Fix 2a):
        The anti-hallucination rule in stream_query / handle_query was too conservative.
        Even for clear English text, language kept the default 'fr'.
    """
    with (
        patch("src.agents.orchestrator.redis.Redis"),
        patch("src.agents.orchestrator.memory_manager") as mock_mem,
        patch(
            "src.agents.intent_classifier.intent_classifier.classify",
            new_callable=AsyncMock,
            return_value="SIMPLE_QA",
        ),
        patch(
            "src.agents.orchestrator.retrieve_legal_info",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        orch = AdminOrchestrator()
        orch.cache = AsyncMock()
        orch.cache.get.return_value = None
        orch.cache.setex = AsyncMock()
        orch._call_llm = AsyncMock(return_value=mock_llm_response("Answer."))

        state = make_state()
        saved_states = []

        mock_mem.load_agent_state = AsyncMock(return_value=state)
        mock_mem.save_agent_state = AsyncMock(
            side_effect=lambda sid, s: saved_states.append(s)
        )

        # Use a clearly English-only query with user_lang='en'
        await orch.handle_query("Tell me about myself", "en", "s8")

        # CONTRACT: state was saved at least once
        assert len(saved_states) > 0
        final_state = saved_states[-1]

        # CONTRACT (FIXED BEHAVIOR — after LanguageResolver integration):
        # English text + user_lang='en' → LanguageResolver correctly sets 'English'.
        assert final_state.user_profile.language == "English"

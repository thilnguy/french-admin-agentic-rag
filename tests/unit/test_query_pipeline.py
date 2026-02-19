"""
Unit tests for QueryPipeline.

All external dependencies (goal_extractor, query_rewriter, intent_classifier,
profile_extractor) are injected as mocks. No LLM calls, no network, no imports
of orchestrator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from src.shared.query_pipeline import QueryPipeline, PipelineResult


def make_pipeline(
    goal_result="Obtenir un permis de conduire",
    rewrite_result="How to get a driving license in France as a Vietnamese resident",
    intent_result="SIMPLE_QA",
    profile_result=None,
):
    """Build a QueryPipeline with configurable mock responses."""
    goal_extractor = MagicMock()
    goal_extractor.extract_goal = AsyncMock(return_value=goal_result)

    query_rewriter = MagicMock()
    query_rewriter.rewrite = AsyncMock(return_value=rewrite_result)

    intent_classifier = MagicMock()
    intent_classifier.classify = AsyncMock(return_value=intent_result)

    profile_extractor = MagicMock()
    profile_extractor.extract = AsyncMock(return_value=profile_result or {})

    return QueryPipeline(
        goal_extractor=goal_extractor,
        query_rewriter=query_rewriter,
        intent_classifier=intent_classifier,
        profile_extractor=profile_extractor,
    )


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_returns_result_dataclass():
    pipeline = make_pipeline()
    result = await pipeline.run(query="How do I get a license?", chat_history=[])
    assert isinstance(result, PipelineResult)


@pytest.mark.asyncio
async def test_pipeline_propagates_rewritten_query():
    pipeline = make_pipeline(rewrite_result="driving license France for Vietnamese")
    result = await pipeline.run(query="How do I get a license?", chat_history=[])
    assert result.rewritten_query == "driving license France for Vietnamese"


@pytest.mark.asyncio
async def test_pipeline_propagates_intent():
    pipeline = make_pipeline(intent_result="COMPLEX_PROCEDURE")
    result = await pipeline.run(query="I need help applying", chat_history=[])
    assert result.intent == "COMPLEX_PROCEDURE"


@pytest.mark.asyncio
async def test_pipeline_propagates_new_core_goal():
    pipeline = make_pipeline(goal_result="Renouveler un titre de séjour")
    result = await pipeline.run(
        query="I want to renew my card", chat_history=[], current_goal=None
    )
    assert result.new_core_goal == "Renouveler un titre de séjour"


@pytest.mark.asyncio
async def test_pipeline_propagates_extracted_data():
    data = {"language": "en", "nationality": "Americaine"}
    pipeline = make_pipeline(profile_result=data)
    result = await pipeline.run(query="I am American", chat_history=[])
    assert result.extracted_data == data


# ---------------------------------------------------------------------------
# Goal locking: existing goal is passed to goal_extractor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_passes_current_goal_to_extractor():
    pipeline = make_pipeline(goal_result="Obtenir un permis de conduire")
    goal_call_args = []

    async def capturing_extract_goal(query, history, current_goal):
        goal_call_args.append(current_goal)
        return "Obtenir un permis de conduire"

    pipeline._goal_extractor.extract_goal = capturing_extract_goal

    await pipeline.run(
        query="I have my titre de séjour",
        chat_history=[],
        current_goal="Obtenir un permis de conduire",
    )
    assert goal_call_args[0] == "Obtenir un permis de conduire"


@pytest.mark.asyncio
async def test_pipeline_passes_user_profile_to_rewriter():
    pipeline = make_pipeline()
    rewrite_call_kwargs = {}

    async def capturing_rewrite(query, history, core_goal=None, user_profile=None):
        rewrite_call_kwargs["core_goal"] = core_goal
        rewrite_call_kwargs["user_profile"] = user_profile
        return query

    pipeline._query_rewriter.rewrite = capturing_rewrite

    profile_dict = {"language": "fr", "nationality": "Vietnamienne"}
    await pipeline.run(
        query="J'ai une carte", chat_history=[], user_profile_dict=profile_dict
    )
    assert rewrite_call_kwargs["user_profile"] == profile_dict


# ---------------------------------------------------------------------------
# Contextual continuation ("Yes Trap" fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contextual_continuation_short_answer_after_question():
    """Short answer after agent question → COMPLEX_PROCEDURE, no intent_classifier call."""
    pipeline = make_pipeline(intent_result="SIMPLE_QA")
    pipeline._intent_classifier.classify = AsyncMock(return_value="SIMPLE_QA")

    history = [
        HumanMessage(content="I want to drive in France"),
        AIMessage(content="Do you have a valid visa?"),
    ]

    result = await pipeline.run(query="Yes", chat_history=history)
    assert result.intent == "COMPLEX_PROCEDURE"
    assert result.is_contextual_continuation is True
    # Intent classifier should NOT be called when short-circuiting
    pipeline._intent_classifier.classify.assert_not_called()


@pytest.mark.asyncio
async def test_contextual_continuation_false_for_long_answer():
    """Long answer should NOT trigger contextual continuation."""
    history = [
        HumanMessage(content="I want a driving license"),
        AIMessage(content="Do you have your residence permit?"),
    ]

    pipeline = make_pipeline(intent_result="SIMPLE_QA")
    result = await pipeline.run(
        query="Yes I have my residence permit from last year and it expires soon",
        chat_history=history,
    )
    assert result.is_contextual_continuation is False


@pytest.mark.asyncio
async def test_contextual_continuation_false_without_question():
    """No prior question in history → no short-circuit even for short answer."""
    history = [
        HumanMessage(content="I want a driving license"),
        AIMessage(content="I can help you with that."),
    ]

    pipeline = make_pipeline(intent_result="SIMPLE_QA")
    result = await pipeline.run(query="Yes", chat_history=history)
    assert result.is_contextual_continuation is False


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_goal_extraction_failure_falls_back():
    """If goal extraction raises, pipeline continues with current_goal unchanged."""
    pipeline = make_pipeline()
    pipeline._goal_extractor.extract_goal = AsyncMock(
        side_effect=RuntimeError("LLM timeout")
    )

    result = await pipeline.run(
        query="I need help",
        chat_history=[],
        current_goal="Existing Goal",
    )
    assert result.new_core_goal == "Existing Goal"
    assert isinstance(result.rewritten_query, str)


@pytest.mark.asyncio
async def test_pipeline_rewrite_failure_falls_back_to_original():
    """If query rewriting raises, falls back to original query."""
    pipeline = make_pipeline()
    pipeline._query_rewriter.rewrite = AsyncMock(side_effect=RuntimeError("timeout"))

    result = await pipeline.run(query="I need a license", chat_history=[])
    assert result.rewritten_query == "I need a license"


@pytest.mark.asyncio
async def test_pipeline_intent_failure_returns_unknown():
    """If intent classification raises, returns 'UNKNOWN'."""
    pipeline = make_pipeline()
    pipeline._intent_classifier.classify = AsyncMock(
        side_effect=RuntimeError("failure")
    )

    history = [AIMessage(content="What can I help you with?")]
    # Long query to avoid contextual continuation short-circuit
    result = await pipeline.run(
        query="I want to apply for a French driving license as a Vietnamese resident",
        chat_history=history,
    )
    assert result.intent == "UNKNOWN"


@pytest.mark.asyncio
async def test_pipeline_profile_extraction_failure_returns_empty():
    """If profile extraction raises, returns empty dict."""
    pipeline = make_pipeline()
    pipeline._profile_extractor.extract = AsyncMock(
        side_effect=RuntimeError("parse error")
    )

    result = await pipeline.run(query="I am Vietnamese", chat_history=[])
    assert result.extracted_data == {}

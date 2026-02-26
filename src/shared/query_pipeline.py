"""
QueryPipeline — preprocessing pipeline extracted from AdminOrchestrator.

PURPOSE:
    Encapsulates the shared preprocessing sequence (goal extraction, query rewrite,
    intent classification, profile extraction) that was previously duplicated between
    `handle_query` (L85-170) and `stream_query` (L458-470).

DIVERGENCES RESOLVED:
    handle_query:
        - Calls goal_extractor before rewriting (anchors to core_goal).
        - Rewrites with core_goal and full user_profile dict.
        - Has contextual-continuation short-circuit ("Yes Trap" fix).

    stream_query:
        - Does NOT call goal_extractor (missing feature).
        - Rewrites without core_goal or user_profile enrichment.
        - Does NOT have the contextual-continuation short-circuit.

    QueryPipeline takes the best of both: goal extraction, enriched rewrite,
    and contextual-continuation detection are always applied.

DESIGN:
    Dataclass output (PipelineResult) for clear typing.
    Each step can be easily mocked in tests.
    No side effects on AgentState — caller is responsible for updating state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class PipelineResult:
    """Results from the preprocessing pipeline. All fields are safe to use directly."""

    rewritten_query: str
    intent: str
    extracted_data: dict = field(default_factory=dict)
    new_core_goal: str | None = None
    is_contextual_continuation: bool = False


class QueryPipeline:
    """
    Stateless preprocessing pipeline for user queries.

    Steps:
        1. Goal extraction (updates core_goal on state)
        2. Query rewriting (anchored to core_goal + user profile)
        3. Contextual continuation detection (the "Yes Trap" fix)
        4. Intent classification
        5. Profile extraction (for language and entity memory)
    """

    def __init__(
        self,
        goal_extractor,
        query_rewriter,
        intent_classifier,
        profile_extractor,
    ):
        self._goal_extractor = goal_extractor
        self._query_rewriter = query_rewriter
        self._intent_classifier = intent_classifier
        self._profile_extractor = profile_extractor

    async def run(
        self,
        query: str,
        chat_history: list,
        current_goal: str | None = None,
        user_profile_dict: dict | None = None,
        model_override: str | None = None,
    ) -> PipelineResult:
        """
        Execute all preprocessing steps and return a PipelineResult.

        Args:
            query: Raw user query.
            chat_history: List of HumanMessage / AIMessage from state.messages.
            current_goal: state.core_goal (may be None on first turn).
            user_profile_dict: state.user_profile.model_dump(exclude_none=True).

        Returns:
            PipelineResult with all preprocessing outputs.
        """
        # Step 1: Goal extraction — must happen BEFORE rewriting
        new_core_goal = current_goal
        try:
            extracted_goal = await self._goal_extractor.extract_goal(
                query, chat_history, current_goal, model_override=model_override
            )
            if extracted_goal and extracted_goal != current_goal:
                new_core_goal = extracted_goal
                logger.info(f"QueryPipeline: Core goal → {new_core_goal}")
        except Exception as e:
            logger.error(f"QueryPipeline: Goal extraction failed: {e}")

        # Step 2: Query rewriting anchored to goal + profile
        try:
            rewritten_query = await self._query_rewriter.rewrite(
                query,
                chat_history,
                core_goal=new_core_goal,
                user_profile=user_profile_dict,
                model_override=model_override,
            )
            logger.info(f"QueryPipeline: Rewritten → {rewritten_query}")
        except Exception as e:
            logger.error(f"QueryPipeline: Query rewrite failed: {e}")
            rewritten_query = query

        # Step 3: Contextual continuation detection ("Yes Trap" fix)
        # Short answers after an agent question are always routed to the AgentGraph.
        is_short_answer = len(query.split()) <= 5
        last_msg_is_question = (
            chat_history
            and chat_history[-1].type == "ai"
            and "?" in chat_history[-1].content
        )
        has_clarification_step = (
            False  # CALLER must inject current_step == CLARIFICATION
        )
        is_contextual_continuation = (
            has_clarification_step or last_msg_is_question
        ) and is_short_answer

        # Step 4: Intent classification (or short-circuit for contextual continuation)
        if is_contextual_continuation:
            intent = "COMPLEX_PROCEDURE"
            logger.info(
                "QueryPipeline: Short-circuit → COMPLEX_PROCEDURE (contextual continuation)"
            )
        else:
            try:
                intent = await self._intent_classifier.classify(rewritten_query, model_override=model_override)
                logger.info(f"QueryPipeline: Intent → {intent}")
            except Exception as e:
                logger.error(f"QueryPipeline: Intent classification failed: {e}")
                intent = "UNKNOWN"

        # Step 5: Profile extraction (always on original query for clean language signal)
        try:
            extracted_data = await self._profile_extractor.extract(query, chat_history, model_override=model_override)
        except Exception as e:
            logger.error(f"QueryPipeline: Profile extraction failed: {e}")
            extracted_data = {}

        return PipelineResult(
            rewritten_query=rewritten_query,
            intent=intent,
            extracted_data=extracted_data,
            new_core_goal=new_core_goal,
            is_contextual_continuation=is_contextual_continuation,
        )


# ---------------------------------------------------------------------------
# Factory: build from real singletons (used by orchestrator)
# ---------------------------------------------------------------------------


def get_query_pipeline() -> QueryPipeline:
    """Build QueryPipeline using production singletons. Call lazily (avoids import cycles)."""
    from src.agents.intent_classifier import intent_classifier
    from src.agents.preprocessor import (
        goal_extractor,
        profile_extractor,
        query_rewriter,
    )

    return QueryPipeline(
        goal_extractor=goal_extractor,
        query_rewriter=query_rewriter,
        intent_classifier=intent_classifier,
        profile_extractor=profile_extractor,
    )

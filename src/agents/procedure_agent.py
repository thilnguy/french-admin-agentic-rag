from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from src.config import settings
from src.agents.state import AgentState, UserProfile
from src.utils.llm_factory import get_llm
from skills.legal_retriever.main import retrieve_legal_info
from src.utils.logger import logger
from src.rules.registry import topic_registry
from src.utils import metrics
import time


class ProcedureGuideAgent:
    def __init__(self):
        # We no longer instantiate self.llm globally
        self.registry = topic_registry

        # Step Analyzer: Determines the current stage of the procedure
        # Uses topic registry's default_step instead of hardcoded topic lists.
        self.step_analyzer_prompt = ChatPromptTemplate.from_template(
            """You are a French Administrative Procedure Guide.
            Analyze the conversation to determine the next step.

            User Query: {query}
            Profile: {user_profile}
            History: {history}
            Topic: {topic_name} (default step: {default_step})

            Possible Steps:
            1. CLARIFICATION: The procedure has CONDITIONAL BRANCHES based on user profile.
               - Use when key variables are MISSING (see below).
               - DO NOT use if the profile already has all needed info.
               - DO NOT use if the query is a direct answer to a previous agent question.

            2. RETRIEVAL: Truly fact-based questions with a SINGLE universal answer.
               - Examples: "How much does a passport cost?", "Can a student work?"
               - RULES FOR COSTS: "How much is X?" is ALWAYS RETRIEVAL.

            3. EXPLANATION: We have the procedure content and profile is complete.
            4. COMPLETED: Procedure finished.

            Missing variables for this topic:
            {missing_variables}

            CRITICAL RULE: When in doubt, choose CLARIFICATION (unless factual/cost question).
            Return ONLY the step name."""
        )


    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )
    async def _run_chain(self, chain, input_data, model_name: str = "unknown"):
        """Wrapper for LCEL chain invocations with retry."""
        start_time = time.time()
        result = await chain.ainvoke(input_data)
        duration = time.time() - start_time
        metrics.LLM_REQUEST_DURATION.labels(model=model_name).observe(duration)
        return result

    async def run(self, query: str, state: AgentState) -> str:
        logger.info(f"ProcedureGuideAgent started for query: {query}")
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.messages[-5:]])

        # DEFENSIVE: use getattr for backward compat with any cached state objects
        core_goal = getattr(state, "core_goal", None)

        # GOAL LOCK: Use core_goal for retrieval to prevent topic drift.
        retrieval_query = core_goal or query
        logger.info(f"ProcedureAgent retrieval anchored to: {retrieval_query}")

        # Detect topic from registry (used by step analyzer and prompt building)
        detected_topic = self.registry.detect_topic(query, getattr(state, "intent", None))
        state.metadata["detected_topic"] = detected_topic
        logger.info(f"ProcedureAgent detected topic: {detected_topic}")

        import asyncio

        step_task = self._determine_step(
            query, state.user_profile.model_dump(), history_str, detected_topic, state=state
        )
        docs_task = retrieve_legal_info(retrieval_query, domain="procedure")

        next_step, docs = await asyncio.gather(step_task, docs_task)
        logger.info(f"Determined next step: {next_step}")
        state.current_step = next_step
        state.retrieved_docs = docs  # Store docs for Hallucination Check

        if next_step == "CLARIFICATION":
            return await self._ask_clarification(query, state, docs)

        # Pre-Synthesis Verification (Groundedness Check)
        is_grounded = await self._verify_groundedness(query, docs, state.user_profile.model_dump(), state=state)
        if not is_grounded:
            logger.warning(f"Groundedness check failed for query: {query}. Falling back to CLARIFICATION.")
            # Inject a system prompt note to force a fallback question
            state.metadata["groundedness_failed"] = True
            return await self._ask_clarification(query, state, docs)

        # For RETRIEVAL or EXPLANATION or default w/ docs
        return await self._explain_procedure(query, state, docs)

    async def _verify_groundedness(self, query: str, docs: List[Dict], user_profile: dict, state: AgentState = None) -> bool:
        """
        Fast check to ensure the retrieved context is actually relevant to the query.
        Prevents hallucinated answers when RAG retrieval fails.
        """
        if not docs:
            return False

        model_override = state.metadata.get("model") if state else None
        fast_llm = get_llm(temperature=0, model_override=model_override)

        context_summary = "\n".join([d["content"][:500] for d in docs[:3]])
        
        prompt = ChatPromptTemplate.from_template(
            """Evaluate if the provided Context contains sufficient information to answer the User Query for the given User Profile.

            User Query: {query}
            User Profile: {profile}
            
            Context:
            {context}

            Rules:
            - Provide ONLY "YES" if the context directly addresses the core administrative task requested.
            - Provide ONLY "NO" if the context is about a completely different procedure, explicitly excludes the user's profile conditions, or is just irrelevant generic information.

            Evaluation (YES/NO):"""
        )
        
        chain = prompt | fast_llm | StrOutputParser()
        try:
            result = await chain.ainvoke({
                "query": query,
                "profile": user_profile,
                "context": context_summary
            })
            return "YES" in result.upper()
        except Exception as e:
            logger.error(f"Groundedness check failed: {e}. Defaulting to True to avoid blocking.")
            return True

    async def _determine_step(
        self, query: str, user_profile: dict, history: str, topic_key: str = "daily_life", state: AgentState = None
    ) -> str:
        topic_rules = self.registry.get_rules(topic_key)
        missing = topic_rules.get_missing_variables(user_profile) if topic_rules else []
        missing_str = topic_rules.format_variable_list(missing) if topic_rules and missing else "All variables known."

        model_override = state.metadata.get("model") if state else None
        llm = get_llm(temperature=0.2, streaming=True, model_override=model_override)

        chain = (self.step_analyzer_prompt | llm | StrOutputParser()).with_config(
            {"tags": ["internal"]}
        )
        return await self._run_chain(
            chain, {
                "query": query,
                "user_profile": user_profile,
                "history": history,
                "topic_name": topic_rules.display_name if topic_rules else "General",
                "default_step": topic_rules.default_step if topic_rules else "CLARIFICATION",
                "missing_variables": missing_str,
            },
            model_name=getattr(llm, "model_name", "unknown")
        )

    async def _ask_clarification(
        self, query: str, state: AgentState, docs: List[Dict]
    ) -> str:
        # If Groundedness Check failed, we explicitly ignore context to avoid hallucinations.
        groundedness_failed = state.metadata.get("groundedness_failed", False)
        
        context_summary = ""
        if docs and not groundedness_failed:
            context_summary = "\n".join([d["content"][:1500] for d in docs[:3]])

        # Get topic-specific rules from registry
        topic_key = state.metadata.get("detected_topic", "daily_life")
        topic_fragment = self.registry.build_prompt_fragment(
            topic_key, state.user_profile.model_dump(), query
        )
        global_rules = self.registry.build_global_rules_fragment()
        
        fallback_instruction = ""
        if groundedness_failed:
            fallback_instruction = """
            CRITICAL INSTRUCTION: The retrieved documents DO NOT MATCH the user's situation. 
            DO NOT answer the user's question, as you lack the legal context. 
            Instead, formulate a [DEMANDER] block asking for the exact name of the administrative document they are trying to process, and explain in [EXPLIQUER] that you need more precision to find the right procedure.
            """

        prompt = ChatPromptTemplate.from_template(
            """{persona}
            User Query: {query}
            Context from official documents:
            {context}
            User Profile (already known): {profile}

            {topic_rules}

            {global_rules}
            
            {fallback_instruction}
            
            Respond in {user_language}.
            """
        )
        model_override = state.metadata.get("model") if state else None
        llm = get_llm(temperature=0.2, streaming=True, model_override=model_override)

        chain = (prompt | llm | StrOutputParser()).with_config(
            {"tags": ["final_answer"]}
        )
        return await self._run_chain(
            chain,
            {
                "query": query,
                "profile": state.user_profile.model_dump(),
                "context": context_summary,
                "user_language": state.user_profile.language or "fr",
                "topic_rules": topic_fragment,
                "global_rules": global_rules,
                "fallback_instruction": fallback_instruction,
                "persona": self.registry.persona,
            },
            model_name=getattr(llm, "model_name", "unknown")
        )

    async def _explain_procedure(
        self, query: str, state: AgentState, docs: List[Dict]
    ) -> str:
        if not docs:
            return "Je ne trouve pas de procédure correspondant exactement à votre demande sur service-public.fr."

        context = "\n\n".join([d["content"][:2000] for d in docs])

        # Get topic-specific rules from registry
        topic_key = state.metadata.get("detected_topic", "daily_life")
        topic_fragment = self.registry.build_prompt_fragment(
            topic_key, state.user_profile.model_dump(), query
        )
        global_rules = self.registry.build_global_rules_fragment()

        prompt = ChatPromptTemplate.from_template(
            """{persona}
            User Query: {query}
            User Location: {user_location}
            Context from official documents:
            {context}

            {topic_rules}

            {global_rules}
            
            Respond in {user_language}.
            """
        )
        model_override = state.metadata.get("model") if state else None
        llm = get_llm(temperature=0.2, streaming=True, model_override=model_override)

        chain = (prompt | llm | StrOutputParser()).with_config(
            {"tags": ["final_answer"]}
        )
        return await self._run_chain(
            chain,
            {
                "query": query,
                "context": context,
                "user_language": state.user_profile.language or "fr",
                "user_location": state.user_profile.location or "votre département",
                "topic_rules": topic_fragment,
                "global_rules": global_rules,
                "persona": self.registry.persona,
            },
            model_name=getattr(llm, "model_name", "unknown")
        )


# Singleton
procedure_agent = ProcedureGuideAgent()

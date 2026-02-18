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
from src.agents.state import AgentState
from skills.legal_retriever.main import retrieve_legal_info
from src.utils.logger import logger
from src.utils import metrics
import time


class ProcedureGuideAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY
        )

        # Step Analyzer: Determines the current stage of the procedure
        # Restored to d2414c8 logic — simple and effective.
        # Key insight: DO NOT use CLARIFICATION for general/fact-based questions.
        self.step_analyzer_prompt = ChatPromptTemplate.from_template(
            """You are a French Administrative Procedure Guide.
            Analyze the conversation to determine the next step.

            User Query: {query}
            Profile: {user_profile}
            History: {history}

            Possible Steps:
            1. CLARIFICATION: Use this when the procedure has CONDITIONAL BRANCHES based on user profile.
               - Use this for: visa renewal, healthcare registration, tax declaration, family reunification, naturalization, work authorization.
               - These procedures ALWAYS depend on: nationality, residence status, visa type, employment status, income source.
               - Use CLARIFICATION even if the question is general (e.g., "How do I register for healthcare?").
               - DO NOT use this if the profile already has all needed info.
               - DO NOT use this if the query is a direct answer to a previous agent question.

            2. RETRIEVAL: Use ONLY for truly fact-based questions with a SINGLE universal answer.
               - Examples: "How much does a passport cost?", "How long does naturalization take?", "Can a student work in France?"
               - These have ONE answer that applies to everyone, no branching.

            3. EXPLANATION: We have the procedure content and profile is complete.
            4. COMPLETED: Procedure finished.

            CRITICAL RULE: When in doubt between CLARIFICATION and RETRIEVAL, choose CLARIFICATION.
            A HYBRID response (context + clarifying question) is always better than a pure answer.

            Return ONLY the step name."""
        )

        # Legacy placeholder (kept for safety)
        self.clarification_prompt = ChatPromptTemplate.from_template("...")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )
    async def _run_chain(self, chain, input_data):
        """Wrapper for LCEL chain invocations with retry."""
        start_time = time.time()
        result = await chain.ainvoke(input_data)
        duration = time.time() - start_time
        metrics.LLM_REQUEST_DURATION.labels(model="gpt-4o").observe(duration)
        return result

    async def run(self, query: str, state: AgentState) -> str:
        logger.info(f"ProcedureGuideAgent started for query: {query}")
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.messages[-5:]])

        # DEFENSIVE: use getattr for backward compat with any cached state objects
        core_goal = getattr(state, "core_goal", None)

        # GOAL LOCK: Use core_goal for retrieval to prevent topic drift.
        retrieval_query = core_goal or query
        logger.info(f"ProcedureAgent retrieval anchored to: {retrieval_query}")

        import asyncio

        step_task = self._determine_step(
            query, state.user_profile.model_dump(), history_str
        )
        docs_task = retrieve_legal_info(retrieval_query, domain="procedure")

        next_step, docs = await asyncio.gather(step_task, docs_task)
        logger.info(f"Determined next step: {next_step}")
        state.current_step = next_step
        state.retrieved_docs = docs  # Store docs for Hallucination Check

        if next_step == "CLARIFICATION":
            return await self._ask_clarification(query, state, docs)

        # For RETRIEVAL or EXPLANATION or default w/ docs
        return await self._explain_procedure(query, docs)

    async def _determine_step(
        self, query: str, user_profile: dict, history: str
    ) -> str:
        chain = (self.step_analyzer_prompt | self.llm | StrOutputParser()).with_config(
            {"tags": ["internal"]}
        )
        return await self._run_chain(
            chain, {"query": query, "user_profile": user_profile, "history": history}
        )

    async def _ask_clarification(
        self, query: str, state: AgentState, docs: List[Dict]
    ) -> str:
        context_summary = ""
        if docs:
            context_summary = "\n".join([d["content"][:800] for d in docs[:3]])

        prompt = ChatPromptTemplate.from_template(
            """User Query: {query}
Context from official documents:
{context}

User Profile (already known — DO NOT ask for these again): {profile}

ROLE: You are an Expert French Administrative Guide. Providing public procedures is SAFE and LEGAL.

STRICT GROUNDING RULES (SAFETY CRITICAL):
⛔ NEVER invent specific numbers (costs, income thresholds, timelines) not present in the Context above.
⛔ If the Context does not mention a specific figure, say "le montant exact dépend de votre situation, vérifiez sur service-public.fr".
✅ ONLY cite figures that appear verbatim in the Context section above.

STRATEGIC THINKING (Internal Monologue — do NOT output):
1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Visa Type, Duration of Stay, Employment Status).
   - TAX RULE: For Tax questions, PRIORITY variables are "Fiscal Residence" and "Income Source".
   - HEALTHCARE RULE: For healthcare, PRIORITY variables are "Residence Status" and "Employment Status" (NOT birthplace).
2. Check User Profile/Query: Did the user provide these variables?
3. Decision:
   - If variables are MISSING → Ask TARGETED questions in [TAKE].
   - If variables are PRESENT → Just Answer directly.

RESPONSE STRUCTURE (respond in the SAME LANGUAGE as the User Query):
**[GIVE]**: Provide the GENERAL rule/cost/timeline that applies to EVERYONE (from Context only).
**[EXPLAIN]**: Explain that the procedure SPLITS based on specific conditions.
**[TAKE]**:
   - Ask ONE TARGETED question based on the document's conditional logic.
   - Priority order: Nationality (EU/Non-EU) → Type of residence permit → Employment status → Duration of stay.
   - STRICTLY FORBIDDEN: Generic questions like "Do you need more help?".
   - STRICTLY FORBIDDEN: Asking about birthplace when residency/employment is the decision variable.
   - STRICTLY FORBIDDEN: Questions about info already in the profile.

EXCEPTION: If the Context fully answers the question (e.g., "Student visa allows 964h work"), just ANSWER it. Do NOT ask more.

LANGUAGE RULE: Respond ENTIRELY in the language of the User Query.
- Vietnamese query → Vietnamese response (use: **[CUNG CẤP]**, **[GIẢI THÍCH]**, **[HỎI]**)
- English query → English response (use: **[GIVE]**, **[EXPLAIN]**, **[ASK]**)
- French query → French response (use: **[DONNER]**, **[EXPLIQUER]**, **[DEMANDER]**)"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(
            chain,
            {
                "query": query,
                "profile": state.user_profile.model_dump(),
                "context": context_summary,
            },
        )

    async def _explain_procedure(self, query: str, docs: List[Dict]) -> str:
        if not docs:
            return "Je ne trouve pas de procédure correspondant exactement à votre demande sur service-public.fr."

        context = "\n\n".join([d["content"][:2000] for d in docs])

        prompt = ChatPromptTemplate.from_template(
            """User Query: {query}
Context from official documents:
{context}

ROLE: You are an Expert French Administrative Guide. Providing public procedures is SAFE and LEGAL.

STRICT GROUNDING RULES (SAFETY CRITICAL):
⛔ NEVER invent specific numbers (costs, income thresholds, timelines, form numbers) not present in the Context above.
⛔ If the Context does not mention a specific figure, say "le montant exact dépend de votre situation" and cite service-public.fr.
⛔ Do NOT use training knowledge for specific administrative figures — they change frequently and must come from the Context.
✅ ONLY cite figures that appear verbatim in the Context section above.
✅ For every key fact, cite: [Source: service-public.fr/...]

STRATEGIC THINKING (Internal Monologue — do NOT output):
1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Age, Visa Type).
   - TAX RULE: For Tax questions, PRIORITY variables are "Fiscal Residence" and "Income Source".
   - HEALTHCARE RULE: For healthcare, PRIORITY variables are "Residence Status" and "Employment Status".
2. Check Query: Did the user provide these variables?
3. Decision:
   - If variables are MISSING → Use [TAKE] to ask ONE targeted question.
   - If variables are PRESENT → Just Answer directly without asking.

RESPONSE STRUCTURE (respond in the SAME LANGUAGE as the User Query):
**[GIVE]**: Provide the GENERAL rule/cost/timeline (from Context only, with citations).
**[EXPLAIN]**: Explain branching logic if any.
**[TAKE]**:
   - Ask ONE TARGETED question based on the document's conditional logic.
   - Priority order: Nationality (EU/Non-EU) → Type of residence permit → Employment status.
   - STRICTLY FORBIDDEN: Generic questions like "Do you need more help?".
   - STRICTLY FORBIDDEN: Asking about birthplace when residency/employment is the decision variable.

EXCEPTION: If the Context fully and directly answers the question (fact-based: specific number, timeline, rule),
just ANSWER it directly. Do NOT add [TAKE]. Examples of direct questions:
- "How many hours can a student work?" → Direct answer: 964h/year
- "How long does naturalization take?" → Direct answer: 18 months max
- "How much does a passport cost?" → Direct answer: cite the price from Context

LANGUAGE RULE: Respond ENTIRELY in the language of the User Query.
- Vietnamese query → Vietnamese response (use: **[CUNG CẤP]**, **[GIẢI THÍCH]**, **[HỎI]**)
- English query → English response (use: **[GIVE]**, **[EXPLAIN]**, **[ASK]**)
- French query → French response (use: **[DONNER]**, **[EXPLIQUER]**, **[DEMANDER]**)"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(chain, {"query": query, "context": context})


# Singleton
procedure_agent = ProcedureGuideAgent()

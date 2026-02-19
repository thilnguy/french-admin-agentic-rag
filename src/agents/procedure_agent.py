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
               - Use this for: visa renewal, healthcare registration, tax declaration, family reunification, naturalization, DRIVING LICENSE EXCHANGE.
               - These procedures ALWAYS depend on: nationality, residence status, visa type, employment status, income source, foreign license status.
               - Use CLARIFICATION even if the question is general (e.g., "How do I register for healthcare?").
               - DO NOT use this if the profile already has all needed info.
               - DO NOT use this if the query is a direct answer to a previous agent question.

            2. RETRIEVAL: Use ONLY for truly fact-based questions with a SINGLE universal answer (or very minor variations).
               - Examples: "How much does a passport cost?", "How long does naturalization take?", "Can a student work in France?"
               - RULES FOR STUDENTS: "Can I work with a student visa?" is ALWAYS RETRIEVAL (Answer: Yes, 964 hours).
               - RULES FOR COSTS: "How much is X?" is ALWAYS RETRIEVAL.

            3. EXPLANATION: We have the procedure content and profile is complete.
            4. COMPLETED: Procedure finished.

            CRITICAL RULE: When in doubt between CLARIFICATION and RETRIEVAL, choose CLARIFICATION, UNLESS it is about Student Work Rights or Costs.
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
        return await self._explain_procedure(query, state, docs)

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
            # Increased to 1500 to ensure enough context for the summary
            context_summary = "\n".join([d["content"][:1500] for d in docs[:3]])

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
   - FAMILY REUNIFICATION RULE: PRIORITY variables are "Nationality" (EU vs Non-EU) AND "Residence Status" (18 months rule). ASK THESE FIRST.
   - TAX RULE: For Tax, PRIORITY is "Fiscal Residence" implies "Income Source".
   - HEALTHCARE RULE: For healthcare, PRIORITY is "Residence Status" (legal resident > 3 months) and "Employment Status".
   - DRIVING LICENSE: PRIORITY is "Residency Status" and "Foreign License Status".

2. Check User Profile: Match known variables.
   - ⛔ **ANTI-HALLUCINATION**: NEVER assume nationality or status based on language. Vietnam language != Vietnamese nationality.

3. Decision:
   - If variables are MISSING → Ask the MOST CRITICAL missing variable in [DEMANDER].
   - If variables are PRESENT → Just Answer directly.

RESPONSE STRUCTURE (respond in {user_language}):
**[DONNER]** (or equivalent): MANDATORY. Summarize the GENERAL procedure found in the context.
   - You MUST provide this summary BEFORE asking questions.
   - If context is empty, say "I found no specific procedure, but..."

**[EXPLIQUER]** (or equivalent): Explain that the procedure VARIES based on valid conditions.

**[DEMANDER]** (or equivalent):
   - Ask ONE TARGETED question to narrow down the case.
   - Priority order: Nationality (EU/Non-EU) → Residence Status → Employment/Income → Duration of stay.
   - STRICTLY FORBIDDEN: Generic questions.
   - STRICTLY FORBIDDEN: Asking for info already in profile.

LANGUAGE RULE:
- Respond ENTIRELY in {user_language}.
- Use localized tags: **[DONNER]**, **[EXPLIQUER]**, **[DEMANDER]** (e.g. **[CUNG CẤP]**, **[GIẢI THÍCH]**, **[YÊU CẦU]** for Vietnamese).
- ⛔ DO NOT mix languages.
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(
            chain,
            {
                "query": query,
                "profile": state.user_profile.model_dump(),
                "context": context_summary,
                "user_language": state.user_profile.language or "fr",
            },
        )

    async def _explain_procedure(
        self, query: str, state: AgentState, docs: List[Dict]
    ) -> str:
        if not docs:
            return "Je ne trouve pas de procédure correspondant exactement à votre demande sur service-public.fr."

        context = "\n\n".join([d["content"][:2000] for d in docs])

        prompt = ChatPromptTemplate.from_template(
            """User Query: {query}
User Location: {user_location}
Context from official documents:
{context}

ROLE: You are an Expert French Administrative Guide. Providing public procedures is SAFE and LEGAL.

CRITICAL LANGUAGE RULE:
You MUST respond ENTIRELY in {user_language}.
- If {user_language} is Vietnamese -> Respond in VIETNAMESE.
- If {user_language} is English -> Respond in ENGLISH.
- If {user_language} is French -> Respond in FRENCH.
- Do NOT mix languages. Do NOT use French tags like [DONNER] if the user language is Vietnamese.

STRICT GROUNDING RULES (SAFETY CRITICAL):
⛔ NEVER invent specific numbers (costs, income thresholds, timelines, form numbers) not present in the Context above.
⛔ If the Context does not mention a specific figure, say "le montant exact dépend de votre situation" and cite service-public.fr.
✅ ONLY cite figures that appear verbatim in the Context section above.
✅ For every key fact, cite: [Source: service-public.fr/...]

STRATEGIC THINKING (Internal Monologue — do NOT output):
1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Age, Visa Type).
   - FAMILY REUNIFICATION: Priority is "Nationality" AND "Residence Status" (18 months?).
   - TAX RULE: Priority is "Fiscal Residence" AND "Income Source".
   - HEALTHCARE RULE: Priority is "Residence Status" AND "Employment Status".
   - DRIVING LICENSE: Priority is "Residency Status" AND "Foreign License Status".
   - ⛔ **ANTI-HALLUCINATION**: NEVER assume nationality or status based on query language.

2. **DEADLINE ANALYSIS (URGENCY CHECK)**:
   - READ the Context to find specific TIME LIMITS (e.g., "1 year to exchange", "register within 3 months").
   - COMPARE with User Profile "Duration of Stay".
   - **MATH & LOGIC**:
     - Convert both to Months (e.g., 1 year = 12 months).
     - Calculate: Remaining Time = Deadline - Duration of Stay.
     - IF Remaining Time < 3 months: YOU MUST TRIGGER **ALERT MODE**.
   - IF User is OVER the deadline: TRIGGER EXPIRED WARNING.

3. Check Query: Did the user provide these variables?

4. Decision:
   - If variables are MISSING → Use [DEMANDER] to ask ONE targeted question.
   - **CRITICAL**: You MUST explain WHY you are asking.
   - If variables are PRESENT → Just Answer directly.

    **ALLOWED EXTERNAL KNOWLEDGE**:
    - You MAY use your internal knowledge of geography (Vietnam -> Non-EU).
    - If "Nationality" is known, IMMEDIATELY decide if it is EU/EEE or Non-EU.

    **CONTEXT FILTERING**:
    - IGNORE information that explicitly contradicts the User Profile.

RESPONSE STRUCTURE (respond in {user_language}):
**[DONNER]** (or equivalent: [GIVE], [CUNG CẤP]): Provide the GENERAL rule/cost/timeline (from Context only, with citations).
   - **MANDATORY**: If the procedure is online, INJECT the link: `https://permisdeconduire.ants.gouv.fr` (for driving license) or `https://administration-etrangers-en-france.interieur.gouv.fr` (for residence).
   - **PERSONALIZATION**: If User Location is known (e.g. Lyon), mention specific local authorities (e.g. "Préfecture de Lyon", "Cour d'appel de Lyon" for translators).

**[EXPLIQUER]** (or equivalent: [EXPLAIN], [GIẢI THÍCH]): Explain branching logic if any.
   - **ALERT MODE**: If Urgency Check triggered, YOU MUST START this section with "⚠️ **[URGENT]**: Vous avez moins de X mois!" (translated).

**[DEMANDER]** (or equivalent: [ASK], [YÊU CẦU]):
   - Ask ONE TARGETED question based on the document's conditional logic.
   - Priority order: Nationality (EU/Non-EU) → Residence Status → Employment.
   - **STRICTLY FORBIDDEN**: Generic questions like "Do you need more help?".
   - **OPTIONAL**: If no critical variables are missing, you MAY ask the NEXT PRACTICAL STEP (e.g., "Have you prepared the translation?").
   - STRICTLY FORBIDDEN: Asking for info already in the profile.

EXCEPTION: If the Context fully and directly answers the question (fact-based, or same answer for all groups like 'Students can work'), just ANSWER it directly. Do NOT add [DEMANDER].
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(
            chain,
            {
                "query": query,
                "context": context,
                "user_language": state.user_profile.language or "fr",
                "user_location": state.user_profile.location or "votre département",
            },
        )


# Singleton
procedure_agent = ProcedureGuideAgent()

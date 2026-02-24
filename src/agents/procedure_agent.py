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
from src.utils import metrics
import time


class ProcedureGuideAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.2)

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
 
ROLE: You are a French Administration Assistant. Reason step-by-step before answering.
        
STRICT RESPONSE STRUCTURE:
**[DONNER]**: Preliminary status based on current info.
**[EXPLIQUER]**: Why you need more info.
**[DEMANDER]**: You MUST ask for 2-3 specific missing details based on the TOPIC:
   - IMMIGRATION: Nationality, Visa Type, Expiry Date, AND 'Family situation' (for 10-year residency).
   - VISA RENEWALS (Passeport Talent): 'Convention d'accueil' or 'Contract extension proof'.
   - WORK/LABOR: Contract type (CDI/CDD), Proof of hours (for unpaid wages), Company size (mandatory for chômage technique), Region/Line (mandatory for strikes).
   - TAXES: Annual Income, Fiscal residence, Family composition.
   - TRANSPORT/DAILY LIFE: 'Line used' and 'Period of the strike' (for refunds). 'Activity type' (for insurance).
   - SOCIAL/HEALTH: Disability percentage (for AAH), Age, Employment status.
   - FAMILY/BIRTH: Marital status (mandatory for birth registration), Place of birth (mandatory), Urgency/Emergency level (mandatory for lost documents).

STRICT MANDATE: ONLY ask for variables relevant to the detected topic. Do NOT ask for 'Nationality' unless it is an IMMIGRATION query. DO NOT ask conversational questions (e.g., 'Have you talked to your boss?'). Always ask for the technical variables above.
        
STRICT GROUNDING RULES (SAFETY CRITICAL):
⛔ NEVER invent specific numbers (costs, income thresholds, timelines) not present in the Context above.
⛔ If the Context does not mention a specific figure, say "le montant exact dépend de votre situation, vérifiez sur service-public.fr".
✅ ONLY cite figures that appear verbatim in the Context section above.

STRATEGIC THINKING (Internal Monologue):
1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Visa Type, Duration of Stay, Employment Status).
2. Check User Profile: Identify which of these are MISSING.
3. Decision: YOU MUST ASK for 2-3 specific MISSING variables. Generic questions like "Do you need more help?" are FORBIDDEN.
4. EXCEPTION: If the user provides a direct answer to an earlier [DEMANDER], you must proceed to [DONNER] + [EXPLIQUER] for that new info, then ASK the NEXT set of variables.
   - If variables are MISSING → Ask the MOST CRITICAL missing variable in [DEMANDER].
   - If variables are PRESENT → Just Answer directly.

RESPONSE STRUCTURE (respond in {user_language}):
**[DONNER]** (or equivalent): MANDATORY. Summarize the GENERAL procedure found in the context.
   - You MUST provide this summary BEFORE asking questions.
   - If context is empty, say "I found no specific procedure, but..."

**[EXPLIQUER]** (or equivalent): Explain that the procedure VARIES based on valid conditions.

**[DEMANDER]** (or equivalent):
   - Ask 2-3 specific TARGETED questions to narrow down the case.
   - MUST ASK for: Region/Line/Company Size (Labor), Place of Birth/Marital Status (Identity), or Contract Extension (Visas) if applicable to the topic.
   - Do NOT ask for information already in the profile: {profile}
   - STRICTLY FORBIDDEN: Generic questions.

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
        ROLE: You are a French Administration Assistant. Reason step-by-step before answering.
        
        STRICT RESPONSE STRUCTURE:
        **[DONNER]**: The main procedure steps.
        **[EXPLIQUER]**: Details, documents, and legal criteria.
        **[DEMANDER]**: Mandatory clarification (e.g., specific location or user's next availability).
        
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
   - TAX RULE: Priority is "Residence Status" AND "Income Sources". You MUST ask for these if missing.
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
   - Ask 2-3 TARGETED questions based on the document's conditional logic.
   - MANDATORY SUB-TOPIC VARIABLES:
     * Labor/Strike: Region, Company Size, Affect Line.
     * Identity/Birth: Place of Birth, Marital Status of parents.
     * Immigration/Passeport Talent: New Convention d'accueil status.
   - STRICTLY FORBIDDEN: Generic questions like "Do you need more help?".
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

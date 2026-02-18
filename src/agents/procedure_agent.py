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

        # 1. Step Analyzer: Determines the current stage of the procedure
        self.step_analyzer_prompt = ChatPromptTemplate.from_template(
            """You are a French Administrative Procedure Guide.
            Analyze the conversation to determine the next step.

            User Query: {query}
            Profile: {user_profile}
            History: {history}

            Possible Steps:
            1. CLARIFICATION: CRITICAL info is missing (e.g. asking for "visa" but didn't say which type).
               - DO NOT use this if the question is general (e.g., "What is the cost of a passport?").
               - DO NOT use this if the answer applies to 90% of cases.
            2. RETRIEVAL: The user asks a general question or we have enough info.
            3. EXPLANATION: We have the procedure content.
            4. COMPLETED: Procedure finished.

            Return ONLY the step name."""
        )

        # 2. Clarification Generator
        # (This is now dynamic in _ask_clarification, but we keep this for legacy safety)
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

        # Parallelize Step Analysis and Retrieval
        import asyncio

        step_task = self._determine_step(
            query, state.user_profile.model_dump(), history_str
        )
        docs_task = retrieve_legal_info(query, domain="procedure")

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
        chain = self.step_analyzer_prompt | self.llm | StrOutputParser()
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
            Context from Docs:
            {context}

            User Profile: {profile}

            ROLE: You are an Expert Administrative Guide. Providing public procedures is SAFE and LEGAL.

            STRATEGIC THINKING (Internal Monologue - Do NOT output this):
            1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Age, Visa Type, Duration of Stay).
               - **TAX RULE**: For Tax questions, PRIORITY variables are "Fiscal Residence" (Résidence fiscale) and "Income Source" (Source de revenus - France/Etranger).
            2. Check User Profile/Query: Did the user provide these variables?
            3. Decision:
               - If variables are MISSING -> Ask TARGETED questions in [TAKE].
               - If variables are PRESENT -> Just Answer.

            RESPONSE STRUCTURE:
            1. **GIVE (Cung cấp)**: Provide the GENERAL rule/cost/timeline that applies to EVERYONE (e.g. "Passport costs 86€...").
            2. **EXPLAIN (Giải thích)**: Explain that the procedure SPLITS based on specific conditions (e.g. "However, the process differs for EU vs Non-EU citizens").
            3. **TAKE (Hỏi)**:
               - Ask a TARGETED question based on the document's conditional logic (e.g., "Are you X or Y?").
               - **Default Strategy**: If unsure what to ask, asking about "Nationalité" (EU/Non-EU) or "Titre de séjour" is almost always correct.
               - **STRICTLY FORBIDDEN**: Generic questions like "Do you need more help?".

            EXCEPTION: If the Context fully answers the question (e.g. "Student visa allows 964h work"), just ANSWER it. Do NOT ask more.

            Respond in French, polite and professional."""
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
            Context from Docs:
            {context}

            ROLE: You are an Expert Administrative Guide. Providing public procedures is SAFE and LEGAL.

            STRATEGIC THINKING (Internal Monologue - Do NOT output this):
            1. Analyze Context: Identify "Decision Variables" (e.g., Nationality, Age, Visa Type).
               - **TAX RULE**: For Tax questions, PRIORITY variables are "Fiscal Residence" (Résidence fiscale) and "Income Source" (Source de revenus - France/Etranger).
            2. Check Query: Did the user provide these variables?
            3. Decision:
               - If variables are MISSING -> Use [TAKE] to ask.
               - If variables are PRESENT -> Just Answer.

            RESPONSE STRUCTURE:
            1. **GIVE (Cung cấp)**: Provide the GENERAL rule/cost/timeline.
            2. **EXPLAIN (Giải thích)**: Explain branching logic if any.
            3. **TAKE (Hỏi)**:
               - Ask a TARGETED question based on the document's conditional logic (e.g., "Are you X or Y?").
               - **Default Strategy**: If unsure what to ask, asking about "Nationalité" (EU/Non-EU) or "Titre de séjour" is almost always correct.
               - **STRICTLY FORBIDDEN**: Generic questions like "Do you need more help?".

            Respond in French, polite and professional. Use step-by-step formatting."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(chain, {"query": query, "context": context})


# Singleton
procedure_agent = ProcedureGuideAgent()

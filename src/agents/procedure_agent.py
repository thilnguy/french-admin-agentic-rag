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
            Analyze the conversation history and the user's latest query to determine the next logical step.

            Current known user profile: {user_profile}
            Conversation History: {history}

            Possible Steps:
            1. CLARIFICATION: Essential information is missing (e.g., nationality, age, status) to identify the correct procedure.
            2. RETRIEVAL: We have enough info to find the procedure.
            3. EXPLANATION: We have the procedure content, need to explain the next step to the user.
            4. COMPLETED: The procedure is finished.

            Return ONLY the step name (CLARIFICATION, RETRIEVAL, EXPLANATION, or COMPLETED)."""
        )

        # 2. Clarification Generator
        self.clarification_prompt = ChatPromptTemplate.from_template(
            """You are helping a user with a French administrative procedure.
            You need more information to identify the correct process.

            User Query: {query}
            Missing Info: {missing_info}

            Ask a polite, concise question in French to get this information."""
        )

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

        # Step 1: Analyze Current State/Step
        # For MVP, we might simple-path this, but let's try to be smart.
        # We need to construct a history string
        history_str = "\n".join([f"{m.type}: {m.content}" for m in state.messages[-5:]])

        next_step = await self._determine_step(
            query, state.user_profile.model_dump(), history_str
        )
        logger.info(f"Determined next step: {next_step}")

        state.current_step = next_step

        if next_step == "CLARIFICATION":
            return await self._ask_clarification(query, state)

        elif next_step == "RETRIEVAL":
            # Retrieve docs -> Retrieve usually leads to Explanation
            docs = await retrieve_legal_info(query, domain="procedure")
            # Store docs in metadata or just use them immediately?
            # For this agent, we might want to return the explanation directly.
            # Let's simplify: Retrieval + Explanation in one go for now.
            return await self._explain_procedure(query, docs)

        elif next_step == "EXPLANATION":
            # If we already have context?
            # For MVP, let's treat Retrieval+Explanation as the main active path.
            pass

        # Default Logic (Legacy-style fallback if complex logic fails)
        docs = await retrieve_legal_info(query, domain="procedure")
        return await self._explain_procedure(query, docs)

    async def _determine_step(
        self, query: str, user_profile: dict, history: str
    ) -> str:
        chain = self.step_analyzer_prompt | self.llm | StrOutputParser()
        return await self._run_chain(
            chain, {"query": query, "user_profile": user_profile, "history": history}
        )

    async def _ask_clarification(self, query: str, state: AgentState) -> str:
        # In a real system, we'd identify WHAT is missing.
        # Here we let the LLM generate the question based on the context.
        prompt = ChatPromptTemplate.from_template(
            """User asks: {query}
            We need to Identify the correct administrative procedure.
            Based on: {profile}
            What 1 key piece of info is missing? (e.g. nationality, age?).
            Ask for it in French."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(
            chain, {"query": query, "profile": state.user_profile.model_dump()}
        )

    async def _explain_procedure(self, query: str, docs: List[Dict]) -> str:
        if not docs:
            return "Je ne trouve pas de procédure correspondant exactement à votre demande sur service-public.fr."

        context = "\n\n".join([d["content"][:2000] for d in docs])

        prompt = ChatPromptTemplate.from_template(
            """You are a Guide for French Administration.
            Explain the procedure clearly based on the provided context.
            Use step-by-step formatting (1., 2., 3.).

            Context:
            {context}

            User Question: {query}

            Response (in French):"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return await self._run_chain(chain, {"query": query, "context": context})


# Singleton
procedure_agent = ProcedureGuideAgent()

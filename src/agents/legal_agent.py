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


class LegalResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY
        )
        self.llm_fast = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
        )

        # 1. Query Refiner: Optimizes user query for vector search
        self.refiner_prompt = ChatPromptTemplate.from_template(
            """You are a French Legal Research Assistant.
            Refine the following user query into a precise keyword-based search query for a legal database (service-public.fr, legifrance).
            Remove conversational filler. Focus on administrative terms.

            User Query: {query}

            Refined Query:"""
        )

        # 3. Synthesizer: Generates the final answer
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """You are a rigorous French Administration Assistant.
            Answer the user's question using ONLY the provided context.
            Cite your sources (Service-Public or Legifrance).

            **MANDATORY CITATION RULE**:
            - You MUST cite the source URL for every key fact provided.
            - Use the format: `[Source: service-public.fr/...]` at the end of the distinct section or sentence.
            - Use ONLY sources provided in the Context.

            **LANGUAGE RULE**:
            - You MUST respond ENTIRELY in {user_language}.
            - KEEP official French administrative terms (e.g., 'Titre de séjour', 'Préfecture') in parentheses if there is no direct equivalent, or if the term is essential for identifying the procedure.
            - Example: "You need to apply for a residence permit (Titre de séjour) at the local prefecture (Préfecture)."

            If the provided context does not contain the answer, strictly reply with: "INSUFFICIENT_CONTEXT".

            Context:
            {context}

            Question: {query}

            Answer in {user_language}:"""
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )
    async def _run_chain(self, chain, input_data):
        """Wrapper for LCEL chain invocations with retry."""
        return await chain.ainvoke(input_data)

    async def run(self, query: str, state: AgentState) -> str:
        logger.info(f"LegalResearchAgent started for query: {query}")
        user_lang = state.user_profile.language or "French"

        # Step 1: Refine Query
        refined_query = await self._refine_query(query)
        logger.info(f"Refined query: {refined_query}")

        # Step 2: Search (Iteration 1)
        docs = await retrieve_legal_info(refined_query, domain="general")
        context = self._format_docs(docs)

        # Step 3: Synthesize (with implicit evaluation)
        return await self._synthesize_answer(query, context, user_lang)

    async def _refine_query(self, query: str) -> str:
        # Optimization: Use llm_fast (gpt-4o-mini)
        chain = self.refiner_prompt | self.llm_fast | StrOutputParser()
        return await self._run_chain(chain, {"query": query})

    async def _synthesize_answer(self, query: str, context: str, user_lang: str) -> str:
        if not context:
            return "Je n'ai trouvé aucune information officielle correspondante dans ma base de données."

        chain = self.synthesis_prompt | self.llm | StrOutputParser()
        result = await self._run_chain(
            chain, {"query": query, "context": context, "user_language": user_lang}
        )

        if "INSUFFICIENT_CONTEXT" in result:
            # Logic for fallback or search loop could go here
            return "Désolé, les documents trouvés ne permettent pas de répondre avec certitude."

        return result

    def _format_docs(self, docs: List[Dict]) -> str:
        return "\n\n".join(
            [
                f"Source: {d.get('source', 'Unknown')}\nTitle: {d.get('metadata', {}).get('title', 'N/A')}\nContent: {d.get('content', '')[:1000]}"
                for d in docs
            ]
        )


# Singleton
legal_agent = LegalResearchAgent()

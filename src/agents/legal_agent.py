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

        # 1. Query Refiner: Optimizes user query for vector search
        self.refiner_prompt = ChatPromptTemplate.from_template(
            """You are a French Legal Research Assistant.
            Refine the following user query into a precise keyword-based search query for a legal database (service-public.fr, legifrance).
            Remove conversational filler. Focus on administrative terms.

            User Query: {query}

            Refined Query:"""
        )

        # 2. Evaluator: Checks if retrieved documents are sufficient
        self.evaluator_prompt = ChatPromptTemplate.from_template(
            """You are evaluating search results for a French administrative query.

            User Query: {query}

            Retrieved Documents:
            {context}

            Are these documents sufficient to answer the query?
            Return ONLY 'YES' or 'NO'.
            """
        )

        # 3. Synthesizer: Generates the final answer
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """You are a rigorous French Administration Assistant.
            Answer the user's question using ONLY the provided context.
            Cite your sources (Service-Public or Legifrance).
            If the context is insufficient, state clearly what is missing.

            Context:
            {context}

            Question: {query}

            Answer (in French):"""
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

        # Step 1: Refine Query
        refined_query = await self._refine_query(query)
        logger.info(f"Refined query: {refined_query}")

        # Step 2: Search (Iteration 1)
        docs = await retrieve_legal_info(refined_query, domain="general")
        context = self._format_docs(docs)

        # Step 3: Evaluate
        sufficient = await self._evaluate_context(query, context)

        if (
            not sufficient and docs
        ):  # Only refine if we found *something* but it wasn't enough?
            # Or if we found nothing, maybe try synonyms?
            # For iteration 1, let's just log. In advanced version, we loop.
            logger.info("Context evaluated as insufficient. (Future: Search Loop)")
            # Fallback: maintain current docs but maybe mark confidence low?
            pass

        # Step 4: Synthesize
        return await self._synthesize_answer(query, context)

    async def _refine_query(self, query: str) -> str:
        chain = self.refiner_prompt | self.llm | StrOutputParser()
        return await self._run_chain(chain, {"query": query})

    async def _evaluate_context(self, query: str, context: str) -> bool:
        if not context:
            return False
        chain = self.evaluator_prompt | self.llm | StrOutputParser()
        result = await self._run_chain(chain, {"query": query, "context": context})
        return "YES" in result.strip().upper()

    async def _synthesize_answer(self, query: str, context: str) -> str:
        if not context:
            return "Je n'ai trouvé aucune information officielle correspondante dans ma base de données."

        chain = self.synthesis_prompt | self.llm | StrOutputParser()
        return await self._run_chain(chain, {"query": query, "context": context})

    def _format_docs(self, docs: List[Dict]) -> str:
        return "\n\n".join(
            [
                f"Source: {d.get('source', 'Unknown')}\nTitle: {d.get('metadata', {}).get('title', 'N/A')}\nContent: {d.get('content', '')[:1000]}"
                for d in docs
            ]
        )


# Singleton
legal_agent = LegalResearchAgent()

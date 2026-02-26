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
from src.utils.llm_factory import get_llm
from src.agents.state import AgentState

from skills.legal_retriever.main import retrieve_legal_info
from src.utils.logger import logger


class LegalResearchAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0, streaming=True)
        self.llm_fast = get_llm(temperature=0)


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
            """You are a French Administration Assistant. Reason step-by-step before answering.
            Answer the user's question using ONLY the provided context.
            Cite your sources (Service-Public or Legifrance).
            
            STRICT RESPONSE STRUCTURE:
            **[DONNER]**: Legal answer or status based on law.
            **[EXPLIQUER]**: Explanation of legal articles or criteria.
            **[DEMANDER]**: Mandatory clarification. 
            
            **CLARIFICATION LOGIC**:
            **CLARIFICATION LOGIC**:
            **CLARIFICATION LOGIC**:
            If info is missing, you MUST ask for 2-3 specific details based on the topic:
            - TAXES: Annual income, fiscal household composition, date of last gift.
            - WORK/LABOR: Contract type (CDI/CDD), Proof of hours (for unpaid wages), Company size (mandatory for chômage technique).
            - TRANSPORT/DAILY LIFE: 'Line used' and 'Period of the strike' (for refunds). 'Activity type' (for insurance).
            - VISA RENEWAL: Convention d'accueil status, contract extension proof, AND 'Family situation' (for 10-year residency).
            - IDENTITY/BIRTH/ID: Place of birth and Marital status of parents, or Urgency/Emergency level (for lost docs).
            - LEGAL: Exact case type (litigation, conseil, etc.), court involved.
            - FAMILY/SUCCESSION: Heirs involved, relationship to deceased.

            STRICT MANDATE: ONLY ask for variables relevant to the detected topic. Do NOT ask for 'Nationality' unless it is an IMMIGRATION query. DO NOT ask conversational questions (e.g., 'Have you talked to your boss?'). Always ask for the technical variables above.

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

        # Note: `query` here is already the goal-anchored, pipeline-rewritten query.
        # Step 1: Search using the pipeline-anchored query
        docs = await retrieve_legal_info(query, domain="general")
        
        # Pre-Synthesis Verification (Groundedness Check)
        is_grounded = await self._verify_groundedness(query, docs, state.user_profile.model_dump())
        if not is_grounded:
            logger.warning(f"Groundedness check failed in LegalAgent for query: {query}. Triggering fallback.")
            return await self._ask_clarification_fallback(query, user_lang)

        context = self._format_docs(docs)

        # Step 2: Synthesize
        return await self._synthesize_answer(query, context, user_lang)

    async def _verify_groundedness(self, query: str, docs: List[Dict], user_profile: dict) -> bool:
        if not docs:
            return False

        context_summary = "\n".join([d["content"][:500] for d in docs[:3]])
        
        prompt = ChatPromptTemplate.from_template(
            """Evaluate if the provided Context contains sufficient legal information to answer the User Query.

            User Query: {query}
            User Profile: {profile}
            
            Context:
            {context}

            Rules:
            - Provide ONLY "YES" if the context contains relevant legal definitions, criteria, or statuses.
            - Provide ONLY "NO" if the context is about a different topic, or is just irrelevant info.

            Evaluation (YES/NO):"""
        )
        
        chain = prompt | self.llm_fast | StrOutputParser()
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

    async def _ask_clarification_fallback(self, query: str, user_lang: str) -> str:
        """Fallback response when retrieved documents are irrelevant."""
        prompt = ChatPromptTemplate.from_template(
            """You are a French Administration Assistant.
            The user asked a legal question but your database search returned IRRELEVANT documents.
            
            DO NOT attempt to answer the legal question.
            
            Provide a response following this structure:
            **[DONNER]**: State clearly that you cannot find the specific law or text for their situation.
            **[EXPLIQUER]**: Explain that you need more keywords or context to search the legal database effectively.
            **[DEMANDER]**: Ask them to provide the specific name of the procedure, document, or situation they are inquiring about.
            
            User's original query: {query}
            
            Respond in {user_language}.
            """
        )
        chain = (prompt | self.llm | StrOutputParser()).with_config({"tags": ["final_answer"]})
        return await self._run_chain(chain, {"query": query, "user_language": user_lang})

    async def _refine_query(self, query: str) -> str:
        # Optimization: Use llm_fast (gpt-4o-mini)
        chain = self.refiner_prompt | self.llm_fast | StrOutputParser()
        return await self._run_chain(chain, {"query": query})

    async def _synthesize_answer(self, query: str, context: str, user_lang: str) -> str:
        if not context:
            return "Je n'ai trouvé aucune information officielle correspondante dans ma base de données."

        chain = (self.synthesis_prompt | self.llm | StrOutputParser()).with_config(
            {"tags": ["final_answer"]}
        )
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

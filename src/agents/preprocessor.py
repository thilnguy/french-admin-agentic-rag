from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.agents.state import UserProfile
from src.config import settings

from src.utils.logger import logger


class QueryRewriter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast & Cheap for rewriting
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a Query De-contextualizer for a French Administration Bot.
            Your task is to rewrite the CURRENT QUERY to be standalone, based on the CONVERSATION HISTORY.

            Rules:
            1. Replace pronouns (it, they, this) with specific entities from history.
            2. If the query is already standalone, return it exactly as is.
            3. Do NOT answer the question. Only rewrite it.
            4. Keep the language of the CURRENT QUERY.

            History:
            {history}

            Current Query: {query}

            Standalone Query:"""
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def rewrite(self, query: str, history: list) -> str:
        """
        Rewrites valid conversational queries.
        If history is empty, returns query as is.
        """
        if not history:
            return query

        # Format history for prompt
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-5:]])

        try:
            return await self.chain.ainvoke({"history": history_str, "query": query})
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return query


# Singleton
query_rewriter = QueryRewriter()


class ProfileExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )
        # We use JsonOutputParser with the Pydantic model
        self.parser = JsonOutputParser(pydantic_object=UserProfile)

        self.prompt = ChatPromptTemplate.from_template(
            """You are a Profile Extractor for a French Administration Bot.
            Extract relevant user information from the conversation history and the latest query.

            Target Fields:
            - language (fr, en, vi)
            - name
            - age
            - nationality (Specific nationality, e.g., Française, Américaine, Vietnamienne)
            - residency_status (e.g., Student, Worker, Retiree)
            - visa_type (e.g., VLS-TS, Carte de Résident)
            - duration_of_stay
            - location (City or Region in France)
            - fiscal_residence (France, Etranger)
            - income_source (France, Etranger, Mixte)

            Rules:
            1. Extract ONLY information clearly stated or implied by the user.
            2. If a field is not mentioned, do NOT include it in the JSON (or set to null).
            3. Return a JSON object matching the schema.

            Conversation History:
            {history}

            Latest Query: {query}

            JSON Output:"""
        )

        self.chain = self.prompt | self.llm | self.parser

    async def extract(self, query: str, history: list) -> dict:
        """
        Extracts user profile fields. Returns a dict of found fields.
        """
        if not history and not query:
            return {}

        # Format history
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-10:]])

        try:
            # Run extraction
            return await self.chain.ainvoke({"history": history_str, "query": query})
        except Exception as e:
            logger.error(f"Profile extraction failed: {e}")
            return {}


profile_extractor = ProfileExtractor()

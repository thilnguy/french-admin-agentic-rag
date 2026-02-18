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
            """You are a Goal-Anchored Query Rewriter for a French Administration Bot.
            Your task is to rewrite the CURRENT QUERY into a precise, standalone search query.

            **CRITICAL RULES**:
            1. CORE GOAL LOCK: If a CORE GOAL is provided, the rewritten query MUST remain anchored to it.
               - Example: Core Goal = "Obtenir un permis de conduire". User says "J'ai un titre de séjour".
               - BAD rewrite: "Renouvellement de carte de séjour"
               - GOOD rewrite: "Permis de conduire en France pour un résident légal vietnamien"
            2. PRONOUN RESOLUTION: Replace pronouns (it, they, this) with specific entities from history.
            3. ENTITY ENRICHMENT: Enrich the query with known user profile facts (nationality, residency).
            4. STANDALONE: The rewritten query must be self-contained for a vector database search.
            5. LANGUAGE: Keep the language of the CURRENT QUERY.
            6. NO ANSWERS: Do NOT answer the question. Only rewrite it.

            Core Goal (if known): {core_goal}

            User Profile (known facts): {user_profile}

            Conversation History (last 5 turns):
            {history}

            Current Query: {query}

            Rewritten Standalone Query:"""
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def rewrite(
        self,
        query: str,
        history: list,
        core_goal: str = None,
        user_profile: dict = None,
    ) -> str:
        """
        Rewrites valid conversational queries, anchored to the core goal.
        If history is empty and no core_goal, returns query as is.
        """
        if not history and not core_goal:
            return query

        # Format history for prompt
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-5:]])
        profile_str = str(user_profile) if user_profile else "Unknown"
        goal_str = core_goal if core_goal else "Not yet determined"

        try:
            return await self.chain.ainvoke(
                {
                    "history": history_str,
                    "query": query,
                    "core_goal": goal_str,
                    "user_profile": profile_str,
                }
            )
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
            - has_legal_residency (boolean: true if user says they live "legally", "régulièrement", "hợp pháp" in France, or has a valid titre de séjour / carte de séjour)
            - visa_type (e.g., VLS-TS, Carte de Résident)
            - duration_of_stay
            - location (City or Region in France)
            - fiscal_residence (France, Etranger)
            - income_source (France, Etranger, Mixte)

            **LOGICAL INFERENCE RULES** (Apply these before extracting):
            - "sống hợp pháp" / "living legally" / "en situation régulière" / "légalement" -> has_legal_residency = true
            - "titre de séjour" / "carte de séjour" / "thẻ cư trú" / "visa valide" -> has_legal_residency = true, visa_type = stated type if mentioned
            - If has_legal_residency = true and residency_status is null -> residency_status = "legal resident"

            Rules:
            1. Extract ONLY information clearly stated or logically implied by the user.
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


class GoalExtractor:
    """Extracts and maintains the user's core goal across the conversation."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a Goal Extractor for a French Administration Bot.
            Identify the user's PRIMARY administrative goal from the conversation.

            Rules:
            1. The goal should be a concise French administrative task (e.g., "Obtenir un permis de conduire", "Renouveler un titre de séjour").
            2. GOAL LOCK: If a CURRENT GOAL is already established, PRESERVE IT unless the user EXPLICITLY says they want to do something completely different.
               - User providing personal info (nationality, residency, documents) does NOT change the goal.
               - User saying "I have a carte de séjour" while discussing driving license = still driving license goal.
            3. Only change the goal if the user says something like "Actually, I want to..." or "Let's talk about..." with a new topic.
            4. If no clear goal is found yet, return null.
            5. Return ONLY the goal string (in French), nothing else. No explanation.

            Current Goal (already established, preserve unless explicitly changed): {current_goal}

            Conversation History:
            {history}

            Current Query: {query}

            Core Goal (or null):"""
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def extract_goal(
        self, query: str, history: list, current_goal: str = None
    ) -> str:
        """
        Extracts or confirms the user's core goal.
        If a goal is already established, it is preserved unless explicitly changed.
        """
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history[-5:]])
        goal_str = current_goal if current_goal else "None"

        try:
            result = await self.chain.ainvoke(
                {
                    "history": history_str,
                    "query": query,
                    "current_goal": goal_str,
                }
            )
            result = result.strip()
            if result.lower() in ["null", "none", ""]:
                return current_goal  # Keep existing goal
            return result
        except Exception as e:
            logger.error(f"Goal extraction failed: {e}")
            return current_goal


goal_extractor = GoalExtractor()

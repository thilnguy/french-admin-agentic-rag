from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger
from src.config import settings


class GuardrailManager:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
        )

    async def validate_topic(
        self, query: str, history: list = None
    ) -> Tuple[bool, str]:
        """
        Ensures the query is related to French administration or law,
        considering conversation context for follow-up questions.
        """
        # Format history for the prompt if it exists
        history_text = "No history available."
        if history:
            # Use .type if available, otherwise class name
            history_text = "\n".join(
                [
                    f"{getattr(msg, 'type', msg.__class__.__name__)}: {msg.content}"
                    for msg in history[-6:]
                ]
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a gatekeeper for Marianne AI, a French Administrative Assistant.
            Your only job is to decide if the query is RELEVANT to French administrative tasks or the current conversation.

            RULES:
            1. If the query is about French procedures, law, public services, or ID documents, it is APPROVED.
            2. If the query is a conversational follow-up or meta-question (e.g., "Why?", "What is my name?"), and there is HISTORY showing a previous administrative discussion, it is APPROVED.
            3. NEW queries about unrelated topics (cooking, celebrities, non-French law) are REJECTED.
            4. Personal introductions (e.g., "My name is...") combined with admin questions are APPROVED.

            Respond only with 'APPROVED' or 'REJECTED: [Short reason in Vietnamese]'.

            HISTORY:
            {history}""",
                ),
                ("user", "{query}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"query": query, "history": history_text})
        logger.debug(f"Guardrail Response: {response}")

        if "APPROVED" in response:
            return True, ""
        return False, response.replace("REJECTED:", "").strip()

    async def check_hallucination(
        self, context: str, answer: str, query: str = "", history: list = None
    ) -> bool:
        """
        Checks if the answer is grounded in the context, history, or the current query.
        """
        history_text = "No history available."
        if history:
            history_text = "\n".join(
                [
                    f"{getattr(msg, 'type', msg.__class__.__name__)}: {msg.content}"
                    for msg in history[-6:]
                ]
            )

        logger.debug(f"Hallucination Check - Query: {query}")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a factual verifier for Marianne AI.
            An answer is SAFE if it is supported by:
            1. The provided CONTEXT (Administrative data).
            2. The conversation HISTORY.
            3. The current user QUERY (e.g., names or details the user just introduced).
            4. Common sense/AI identity (e.g., "I am an AI").
            5. REASONABLE SYNTHESIS: A step-by-step summary or paraphrase of administrative procedures is SAFE, even if not word-for-word from the context, as long as no specific numbers (costs, deadlines, quotas) are invented.
            6. CLARIFYING QUESTIONS: Responses that ask for missing information (nationality, residence status, visa type) are always SAFE.

            An answer is a HALLUCINATION ONLY if it:
            - Invents specific numbers (costs, deadlines, quotas) NOT present in the context.
            - Claims a specific rule applies to the user without any basis in context.

            When in doubt, respond SAFE. False rejections are worse than false approvals for this system.
            Respond strictly with 'SAFE' or 'HALLUCINATION'.""",
                ),
                (
                    "user",
                    "CONTEXT:\n{context}\n\nHISTORY:\n{history}\n\nUSER QUERY: {query}\n\nANSWER:\n{answer}",
                ),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke(
            {
                "context": context,
                "answer": answer,
                "query": query,
                "history": history_text,
            }
        )
        logger.debug(f"Hallucination Response: {response}")

        return "SAFE" in response

    def add_disclaimer(self, answer: str, language: str = "fr") -> str:
        """
        Adds a mandatory legal disclaimer. (Sync is fine here as it's just string manip)
        """
        disclaimers = {
            "fr": "\n\n*Note : Ces informations sont données à titre indicatif. Pour toute décision officielle, veuillez consulter le site service-public.fr ou contacter les autorités compétentes.*",
            "en": "\n\n*Note: This information is for guidance only. For official decisions, please consult service-public.fr or contact the relevant authorities.*",
            "vi": "\n\n*Lưu ý: Thông tin này chỉ mang tính chất tham khảo. Để có quyết định chính thức, vui lòng truy cập service-public.fr hoặc liên hệ cơ quan có thẩm quyền.*",
        }
        return answer + disclaimers.get(language.lower()[:2], disclaimers["fr"])


guardrail_manager = GuardrailManager()

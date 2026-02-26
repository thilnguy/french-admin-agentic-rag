from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger
from src.config import settings
from src.utils.llm_factory import get_llm



from src.utils.tracing import tracer
from opentelemetry import trace

class GuardrailManager:
    def __init__(self):
        # Always use a robust model for Guardrails to prevent false refusals,
        # especially for non-French languages or complex logic.
        # Model is configurable via GUARDRAIL_MODEL setting (default: gpt-4o-mini).
        self.llm = ChatOpenAI(
            model=settings.GUARDRAIL_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        )


    @tracer.start_as_current_span("guardrail_validate_topic")
    async def validate_topic(
        self, query: str, history: list = None
    ) -> Tuple[bool, str]:
        span = trace.get_current_span()
        span.set_attribute("query", query)
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
            1. If the query is about French procedures, law, public services, identity documents, transport (Navigo, SNCF, RATP, driving licenses, vehicle registration, Crit'Air), civil registry (birth/death/marriage/kết hôn/mariage), or legal records (Casier Judiciaire/Criminal Record), it is APPROVED.
            2. If the query is about TAXES in France (Impôts, donations, inheritance/succession, IFI, local taxes, tax-free gifts), it is APPROVED.
            3. If the query is about EDUCATION in France (School registration, Assurance scolaire, student aid, bourses, university admin), it is APPROVED.
            4. If the query involves FOREIGN DOCUMENTS being used for French procedures (e.g., "Can I use my UK license?", "Is my US diploma valid?"), it is APPROVED.
            5. If the query is about LABOR rights, strikes, chômage technique, employer disputes, or natural disaster compensation (floods, etc.), it is APPROVED. This includes Vietnamese queries about wages (lương, tiền lương), work contracts (hợp đồng lao động), or working without a contract.
            6. Housing rights, tenant/landlord disputes, social benefits (CAF, AAH, RSA, Chèque Énergie, Retirement) are APPROVED.
            7. Embassy/consular procedures, visa applications, and administrative certificates are APPROVED.
            8. Conversational follow-ups, meta-questions, or personal introductions in an admin context are APPROVED.
            9. HEALTHCARE and health insurance are ALWAYS APPROVED, including queries in Vietnamese (bảo hiểm y tế, sức khỏe) or English about dual-national/cross-border healthcare coverage.
            10. UNRELATED topics (cooking, celebrities, general sports, non-French law unrelated to residency) are REJECTED.

            CRITICAL: Your job is ONLY to validate the TOPIC, not to answer the question. 
            Do NOT say 'REJECTED: It is not possible to do X'. Only say REJECTED if the topic is completely unrelated to French law or administration.
            When in doubt, respond APPROVED.

            Respond only with 'APPROVED' or 'REJECTED: [Short reason in English]'.

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
        # Extract English reason
        reason = response.replace("REJECTED:", "").strip()
        return False, reason

    @tracer.start_as_current_span("guardrail_check_hallucination")
    async def check_hallucination(
        self, context: str, answer: str, query: str = "", history: list = None
    ) -> bool:
        span = trace.get_current_span()
        span.set_attribute("query", query)
        span.set_attribute("answer_length", len(answer))
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
            5. REASONABLE SYNTHESIS: Paraphrasing procedures, summarizing general requirements, or providing common administrative knowledge NOT explicitly in context (e.g., "you must be 18 to vote", "3-5 years residency for 10-year card") is SAFE.
            6. CLARIFYING QUESTIONS: Responses that ask for missing information are always SAFE.

            An answer is a HALLUCINATION ONLY if it:
            - Invents specific data (EXACT prices like "55.23€", precise office addresses, exact quotas) NOT in context.
            - Directly contradicts the context provided.
            - Gives dangerous or incorrect legal advice that could lead to immediate rejection (e.g., "you don't need a visa" for a non-EU citizen).

            When in doubt, respond SAFE. This is a helpful assistant, not a strict legal validator.
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
        [DEPRECATED by UI] The disclaimer is now handled persistently by the frontend UI.
        Returning the answer unmodified.
        """
        return answer


guardrail_manager = GuardrailManager()

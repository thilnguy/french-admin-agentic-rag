from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger
from src.config import settings

class GuardrailManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)

    async def validate_topic(self, query: str, history: list = None) -> Tuple[bool, str]:
        """
        Ensures the query is related to French administration or law, 
        considering conversation context for follow-up questions.
        """
        # Format history for the prompt if it exists
        history_text = "No history available."
        if history:
            # Use .type if available, otherwise class name
            history_text = "\n".join([f"{getattr(msg, 'type', msg.__class__.__name__)}: {msg.content}" for msg in history[-6:]])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a gatekeeper for Marianne AI, a French Administrative Assistant. 
            Your only job is to decide if the query is RELEVANT to French administrative tasks or the current conversation.
            
            RULES:
            1. If the query is about French procedures, law, public services, or ID documents, it is APPROVED.
            2. If the query is a conversational follow-up or meta-question (e.g., "Why?", "What is my name?"), and there is HISTORY showing a previous administrative discussion, it is APPROVED.
            3. NEW queries about unrelated topics (cooking, celebrities, non-French law) are REJECTED.
            4. Personal introductions (e.g., "My name is...") combined with admin questions are APPROVED.
            
            Respond only with 'APPROVED' or 'REJECTED: [Short reason in Vietnamese]'.
            
            HISTORY:
            {history}"""),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"query": query, "history": history_text})
        logger.debug(f"Guardrail Response: {response}")
        
        if "APPROVED" in response:
            return True, ""
        return False, response.replace("REJECTED:", "").strip()

    async def check_hallucination(self, context: str, answer: str, query: str = "", history: list = None) -> bool:
        """
        Checks if the answer is grounded in the context, history, or the current query.
        """
        history_text = "No history available."
        if history:
            history_text = "\n".join([f"{getattr(msg, 'type', msg.__class__.__name__)}: {msg.content}" for msg in history[-6:]])

        logger.debug(f"Hallucination Check - Query: {query}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a factual verifier for Marianne AI. 
            An answer is SAFE if it is supported by:
            1. The provided CONTEXT (Administrative data).
            2. The conversation HISTORY.
            3. The current user QUERY (e.g., names or details the user just introduced).
            4. Common sense/AI identity (e.g., "I am an AI").
            
            An answer is a HALLUCINATION if it makes up new administrative rules or claims facts not in the context.
            
            Respond strictly with 'SAFE' or 'HALLUCINATION'."""),
            ("user", "CONTEXT:\n{context}\n\nHISTORY:\n{history}\n\nUSER QUERY: {query}\n\nANSWER:\n{answer}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"context": context, "answer": answer, "query": query, "history": history_text})
        logger.debug(f"Hallucination Response: {response}")
        
        return "SAFE" in response

    def add_disclaimer(self, answer: str, language: str = "fr") -> str:
        """
        Adds a mandatory legal disclaimer. (Sync is fine here as it's just string manip)
        """
        disclaimers = {
            "fr": "\n\n*Note : Ces thông tin chỉ mang tính chất tham khảo. Pour toute décision officielle, veuillez consulter le site service-public.fr hoặc liên hệ cơ quan ban ngành có thẩm quyền.*",
            "en": "\n\n*Note: This information is for guidance only. For official decisions, please consult service-public.fr or contact the relevant authorities.*",
            "vi": "\n\n*Lưu ý: Thông tin này chỉ mang tính chất tham khảo. Để có quyết định chính thức, vui lòng truy cập service-public.fr hoặc liên hệ cơ quan có thẩm quyền.*"
        }
        return answer + disclaimers.get(language.lower()[:2], disclaimers["fr"])

guardrail_manager = GuardrailManager()

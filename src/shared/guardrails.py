from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class GuardrailManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def validate_topic(self, query: str) -> Tuple[bool, str]:
        """
        Ensures the query is related to French administration or law.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a gatekeeper for a French Administrative Assistant. 
            Determine if the user's query is related to French administrative procedures, public services, or legislation.
            Respond only with 'APPROVED' or 'REJECTED: [Reason in Vietnamese]'.
            Reject queries about medical advice, general entertainment, or non-French specific legal issues."""),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query})
        
        if "APPROVED" in response:
            return True, ""
        return False, response.replace("REJECTED:", "").strip()

    def check_hallucination(self, context: str, answer: str) -> bool:
        """
        Checks if the answer is grounded in the provided context.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a factual verifier. 
            Does the provided Answer contain information that is NOT present or implied in the Context?
            Respond only with 'SAFE' or 'HALLUCINATION'."""),
            ("user", "Context: {context}\n\nAnswer: {answer}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "answer": answer})
        
        return "SAFE" in response

    def add_disclaimer(self, answer: str, language: str = "fr") -> str:
        """
        Adds a mandatory legal disclaimer.
        """
        disclaimers = {
            "fr": "\n\n*Note : Ces thông tin chỉ mang tính chất tham khảo. Pour toute décision officielle, veuillez consulter le site service-public.fr hoặc liên hệ cơ quan ban ngành có thẩm quyền.*",
            "en": "\n\n*Note: This information is for guidance only. For official decisions, please consult service-public.fr or contact the relevant authorities.*",
            "vi": "\n\n*Lưu ý: Thông tin này chỉ mang tính chất tham khảo. Để có quyết định chính thức, vui lòng truy cập service-public.fr hoặc liên hệ cơ quan có thẩm quyền.*"
        }
        return answer + disclaimers.get(language.lower()[:2], disclaimers["fr"])

guardrail_manager = GuardrailManager()

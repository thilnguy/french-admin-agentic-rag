from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings
from enum import Enum


class Intent(str, Enum):
    SIMPLE_QA = "SIMPLE_QA"
    COMPLEX_PROCEDURE = "COMPLEX_PROCEDURE"
    FORM_FILLING = "FORM_FILLING"
    LEGAL_INQUIRY = "LEGAL_INQUIRY"
    UNKNOWN = "UNKNOWN"


class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast & Cheap for classification
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )

        system_prompt = """You are an intent classifier for a French Administration Assistant.
        Classify the user's query into one of the following categories:

        1. SIMPLE_QA: Simple factual questions about documents, costs, locations, or definitions.
           (e.g., "How much is a passport?", "Where is the prefecture?", "What is a Kbis?")

        2. COMPLEX_PROCEDURE: Questions about multi-step processes, personal situations, or "how-to" guides that require maintaining long context.
           (e.g., "How do I apply for a student visa?", "My application was rejected, what next?", "I need to change my address.")

        3. LEGAL_INQUIRY: Questions asking for specific laws, regulations, or legal text references.
           (e.g., "What implies the Article 12 of Civil Code?", "What is the law regarding subletting?")

        4. FORM_FILLING: Explicit requests to help fill out a specific form.
           (e.g., "Help me fill Cerfa 12345", "What do I put in box 3?")

        SPECIAL RULE FOR CONTEXT:
        - If the user provides a STATEMENT (e.g., "I live in Paris", "I am American", "Married") that answers a previous question in the HISTORY:
          - Classify it as **COMPLEX_PROCEDURE** (so the Procedure Agent can process the answer).
          - Do NOT classify as UNKNOWN.

        Return ONLY the category name. Do not add any explanation.

        HISTORY:
        {history}"""

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{query}")]
        )

        self.chain = self.prompt | self.llm

    async def classify(self, query: str, history: list = None) -> str:
        # Format history
        history_text = "No history."
        if history:
            history_text = "\n".join([f"{getattr(m, 'type', 'msg')}: {m.content}" for m in history[-3:]])

        try:
            response = await self.chain.ainvoke({"query": query, "history": history_text})
            intent = response.content.strip().upper()
            if intent in Intent.__members__:
                return intent
            return Intent.UNKNOWN
        except Exception:
            return Intent.UNKNOWN


# Singleton
intent_classifier = IntentClassifier()

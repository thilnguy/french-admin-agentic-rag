from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings
from enum import Enum


class Intent(str, Enum):
    SIMPLE_QA = "SIMPLE_QA"
    COMPLEX_PROCEDURE = "COMPLEX_PROCEDURE"
    FORM_FILLING = "FORM_FILLING"
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

        3. FORM_FILLING: Explicit requests to help fill out a specific form.
           (e.g., "Help me fill Cerfa 12345", "What do I put in box 3?")

        Return ONLY the category name. Do not add any explanation."""

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{query}")]
        )

        self.chain = self.prompt | self.llm

    async def classify(self, query: str) -> str:
        try:
            response = await self.chain.ainvoke({"query": query})
            intent = response.content.strip().upper()
            if intent in Intent.__members__:
                return intent
            return Intent.UNKNOWN
        except Exception:
            return Intent.UNKNOWN


# Singleton
intent_classifier = IntentClassifier()

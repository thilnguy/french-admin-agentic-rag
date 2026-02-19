from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings

# Singleton LLM — avoid creating a new client per call
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY
        )
    return _llm


async def translate_admin_text(text: str, target_language: str):
    """
    Translates French administrative text into English or Vietnamese,
    ensuring technical terms (e.g., Prefecture, Titre de séjour) are correctly contextually translated.
    target_language: 'English' or 'Vietnamese'
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional administrative translator.
        Your task is to translate the user's text strictly into {target_language}.

        CRITICAL RULES:
        1. **Structure Handling**:
           - **IF** the input uses tags like `**[DONNER]**:`, `**[EXPLIQUER]**:`, you MUST PRESERVE the tags and the markdown format, but TRANSLATE the keywords (e.g., [DONNER] -> [GIVE] in English, [CUNG CẤP] in Vietnamese).
           - **IF** the input is plain text (no tags), just translate it accurately.
        2. **Do NOT Refuse**: Always translate the content, whether it is a question, statement, or instruction.
        3. **Legal Accuracy**: Maintain French administrative terms (e.g., 'Titre de séjour', 'Préfecture') if no exact equivalent exists.
        4. **Tone**: Formal and administrative.""",
            ),
            ("user", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"text": text, "target_language": target_language})


if __name__ == "__main__":
    # Example: asyncio.run(translate_admin_text("Demande de titre de séjour à la préfecture", "Vietnamese"))
    pass

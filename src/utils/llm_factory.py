from langchain_openai import ChatOpenAI
from src.config import settings

def get_llm(temperature: float = 0.2, model_override: str = None, streaming: bool = False, provider_override: str = None):
    """
    Factory function to initialize ChatOpenAI with either OpenAI 
    or a Local LLM backend based on settings.
    """
    # Handle UI Dropdown mappings
    if model_override == "Qwen Finetuned (Local)":
        provider_override = "local"
        model_override = settings.LOCAL_LLM_MODEL
    elif model_override == "GPT-4o":
        provider_override = "openai"
        model_override = settings.OPENAI_MODEL

    provider = provider_override or settings.LLM_PROVIDER

    if provider == "local":
        return ChatOpenAI(
            model=model_override or settings.LOCAL_LLM_MODEL,
            temperature=temperature,
            openai_api_key="local-placeholder",
            base_url=settings.LOCAL_LLM_URL,
            streaming=streaming
        )

    return ChatOpenAI(
        model=model_override or settings.OPENAI_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
        streaming=streaming
    )

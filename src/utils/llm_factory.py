from langchain_openai import ChatOpenAI
from src.config import settings

def get_llm(temperature: float = 0.2, model_override: str = None):
    """
    Factory function to initialize ChatOpenAI with either OpenAI 
    or a Local LLM backend based on settings.
    """
    if settings.LLM_PROVIDER == "local":
        return ChatOpenAI(
            model=model_override or settings.LOCAL_LLM_MODEL,
            temperature=temperature,
            openai_api_key="local-placeholder",
            base_url=settings.LOCAL_LLM_URL,
            model_kwargs={"stop": ["<|im_end|>", "<|endoftext|>"]}
        )
    
    return ChatOpenAI(
        model=model_override or settings.OPENAI_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY
    )

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question")
    language: str = Field(
        "fr", pattern="^(fr|en|vi)$", description="Language code (fr, en, vi)"
    )
    session_id: str = Field("default", min_length=1, description="Session identifier")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent's response")


class VoiceChatResponse(BaseModel):
    user_text: str
    answer_text: str
    audio_url: str

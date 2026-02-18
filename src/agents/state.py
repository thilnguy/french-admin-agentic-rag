from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class UserProfile(BaseModel):
    """Stores persistent user information extracted from conversation."""

    language: str = "fr"  # Default to French
    name: Optional[str] = None
    age: Optional[int] = None
    residency_status: Optional[str] = None  # e.g., "student", "worker", "tourist"
    location: Optional[str] = None  # e.g., "Paris"


class AgentState(BaseModel):
    """
    Unified state object for the French Admin Agent.
    Used to pass context between the Orchestrator and (future) Specialist Agents.
    """

    # Conversation History
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        default_factory=list
    )

    # User Context
    user_profile: UserProfile = Field(default_factory=UserProfile)

    # Workflow State
    session_id: str
    intent: Optional[str] = None  # e.g., "SIMPLE_QA", "COMPLEX_PROCEDURE"
    current_step: Optional[str] = None  # e.g., "ASKING_DOCUMENTS"

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_docs: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Stores docs for Hallucination Check

    class Config:
        arbitrary_types_allowed = True

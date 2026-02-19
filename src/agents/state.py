from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class UserProfile(BaseModel):
    """Stores persistent user information extracted from conversation."""

    language: str = "fr"  # Default to French
    name: Optional[str] = None
    age: Optional[int] = None
    nationality: Optional[str] = None  # e.g., "Française", "Américaine"
    residency_status: Optional[str] = (
        None  # e.g., "student", "worker", "tourist", "titre de séjour"
    )
    has_legal_residency: Optional[bool] = (
        None  # True if user stated they live legally in France
    )
    visa_type: Optional[str] = None  # e.g., "VLS-TS", "Carte de résident"
    duration_of_stay: Optional[str] = None  # e.g., "2 ans", "moins de 3 mois"
    location: Optional[str] = None  # e.g., "Paris", "Antibes"
    fiscal_residence: Optional[str] = None  # e.g., "France", "Etranger"
    income_source: Optional[str] = None  # e.g., "France", "Etranger"
    _reasoning: Optional[str] = None  # Debugging: Why this profile was extracted


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
    core_goal: Optional[str] = None  # The user's primary objective, locked across turns

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_docs: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Stores docs for Hallucination Check

    class Config:
        arbitrary_types_allowed = True

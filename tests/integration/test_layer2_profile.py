import asyncio
from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import AgentState, UserProfile
from src.memory.manager import memory_manager
from src.config import settings

# Disable cache for testing
settings.DEBUG = True

async def test_layer2_profile_extraction():
    print("Testing Layer 2: Profile Extraction...")
    
    session_id = "test_layer2_session"
    
    # 1. Clear previous state by overwriting with fresh state
    fresh_state = AgentState(session_id=session_id)
    await memory_manager.save_agent_state(session_id, fresh_state)
    
    # 2. Run Orchestrator with a query containing entities
    orchestrator = AdminOrchestrator()
    # "I am an American student living in Paris, looking for visa renewal."
    query = "Je suis étudiant américain vivant à Paris, je cherche à renouveler mon visa."
    
    print(f"Sending Query: {query}")
    
    response = await orchestrator.handle_query(query, session_id=session_id)
    
    # 3. Verify State
    final_state = await memory_manager.load_agent_state(session_id)
    profile = final_state.user_profile
    
    print(f"Extracted Profile: {profile}")
    
    # Assertions
    # Note: LLM extraction might vary slightly in casing, so we lowercase.
    assert "américain" in (profile.nationality or "").lower(), f"Nationality mismatch: {profile.nationality}"
    assert "student" in (profile.residency_status or "").lower() or "étudiant" in (profile.residency_status or "").lower(), f"Status mismatch: {profile.residency_status}"
    assert "paris" in (profile.location or "").lower(), f"Location mismatch: {profile.location}"
    
    print("SUCCESS: User Profile was updated with extracted entities.")

if __name__ == "__main__":
    asyncio.run(test_layer2_profile_extraction())

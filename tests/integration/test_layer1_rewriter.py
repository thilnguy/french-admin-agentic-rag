import asyncio
from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import AgentState, UserProfile
from src.memory.manager import memory_manager
from langchain_core.messages import HumanMessage, AIMessage


async def test_layer1_rewriting():
    print("Testing Layer 1: Query Rewriting...")

    session_id = "test_layer1_session"

    # 1. Setup State directly in Redis to simulate history
    state = AgentState(
        session_id=session_id,
        user_profile=UserProfile(language="fr"),
        messages=[
            HumanMessage(content="J'habite à Antibes."),
            AIMessage(content="C'est noté, vous résidez à Antibes."),
            HumanMessage(content="Je veux renouveler mon passeport."),
            AIMessage(
                content="Pour renouveler votre passeport, vous devez aller à la mairie."
            ),
        ],
    )
    await memory_manager.save_agent_state(session_id, state)

    # 2. Run Orchestrator with a vague query
    orchestrator = AdminOrchestrator()
    query = "Quels sont les horaires d'ouverture là-bas ?"  # "là-bas" should refer to "Mairie d'Antibes" or just "Antibes"

    print(f"Sending Query: {query}")

    # We mock the internal components to observe the rewrite,
    # but since rewrite happens internally only logging shows it.
    # We can check the state AFTER execution.
    # The rewritten query is stored in state.metadata["current_query"] by my change.

    response = await orchestrator.handle_query(query, session_id=session_id)

    # 3. Verify State
    final_state = await memory_manager.load_agent_state(session_id)
    rewritten = final_state.metadata.get("current_query")

    print(f"Rewritten Query stored in Metadata: {rewritten}")
    print(f"Final Response: {response}")

    if "Antibes" in rewritten or "mairie" in rewritten.lower():
        print("SUCCESS: Query was rewritten with context.")
    else:
        print("FAILURE: Query was NOT rewritten correctly.")


if __name__ == "__main__":
    asyncio.run(test_layer1_rewriting())

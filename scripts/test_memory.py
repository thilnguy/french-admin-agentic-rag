import asyncio
import uuid
from src.agents.orchestrator import AdminOrchestrator

# Initialize Orchestrator
orchestrator = AdminOrchestrator()
session_id = f"test_session_{uuid.uuid4()}"


async def test_memory():
    print(f"Testing Memory with Session ID: {session_id}")

    # 1. First turn: Introduce name and city
    print(
        "\n[Turn 1] Query: My name is Anh and I come from Hanoi. How do I apply for a French passport?"
    )
    res1 = await orchestrator.handle_query(
        "My name is Anh and I come from Hanoi. How do I apply for a French passport?",
        user_lang="en",
        session_id=session_id,
    )
    print(f"Response (English):\n{res1}")

    # 2. Second turn: Ask about the name introduced in turn 1
    print("\n[Turn 2] Query: Do you remember my name? Answer in Vietnamese.")
    res2 = await orchestrator.handle_query(
        "Do you remember my name? Answer in Vietnamese.",
        user_lang="vi",
        session_id=session_id,
    )
    print(f"Response (Vietnamese):\n{res2}")

    if "Anh" in res2:
        print("\n✅ SUCCESS: Memory works! The agent remembered the name.")
    else:
        print("\n❌ FAILURE: Memory issue. The agent forgot the name.")

    # 3. Third turn: Ask about the city introduced in turn 1
    print("\n[Turn 3] Query: Do you remember my city? Answer in Vietnamese.")
    res3 = await orchestrator.handle_query(
        "Do you remember my city? Answer in Vietnamese.",
        user_lang="vi",
        session_id=session_id,
    )
    print(f"Response (Vietnamese):\n{res3}")

    if "Hà Nội" in res3 or "Hanoi" in res3:
        print("\n✅ SUCCESS: Memory works! The agent remembered the city.")
    else:
        print("\n❌ FAILURE: Memory issue. The agent forgot the city.")


if __name__ == "__main__":
    asyncio.run(test_memory())

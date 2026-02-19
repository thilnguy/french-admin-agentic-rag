import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import AdminOrchestrator
import logging

# Set logging to INFO
logging.basicConfig(level=logging.INFO)


async def test_repro_scenario():
    orchestrator = AdminOrchestrator()
    session_id = f"test_repro_stream_{asyncio.get_event_loop().time()}"

    # Turn 1: Vietnamese
    query1 = "Tôi muốn đổi bằng lái xe nước ngoài sang bằng lái xe Pháp"
    print(f"\nTurn 1 (vi): {query1}")
    resp1 = ""
    async for event in orchestrator.stream_query(
        query1, user_lang="fr", session_id=session_id
    ):
        if event["type"] == "token":
            resp1 += event["content"]
        if event["type"] == "status":
            print(f"Status: {event['content']}")
    print(f"Resp 1 (expect vi): {resp1[:100]}...")

    # Turn 2: English "I am american"
    query2 = "I am american"
    print(f"\nTurn 2 (en): {query2}")
    resp2 = ""
    async for event in orchestrator.stream_query(
        query2, user_lang="fr", session_id=session_id
    ):
        if event["type"] == "token":
            resp2 += event["content"]
        if event["type"] == "status":
            print(f"Status: {event['content']}")

    # Check what state thinks now
    final_state = await orchestrator.memory.load_agent_state(session_id)
    print(f"\nFinal State Language: {final_state.user_profile.language}")
    print(f"Resp 2 (expect en): {resp2[:100]}...")

    # Validation
    is_en = any(
        word.lower() in resp2.lower()
        for word in ["give", "explain", "ask", "american", "driving license"]
    )
    if is_en and "[GIẢI THÍCH]" not in resp2 and "[DONNER]" not in resp2:
        print("\n✅ SUCCESS: Streaming Turn 2 responded in English correctly!")
    else:
        print("\n❌ FAILURE: Streaming Turn 2 did not respond in English correctly.")


if __name__ == "__main__":
    asyncio.run(test_repro_scenario())

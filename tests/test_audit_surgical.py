import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import AdminOrchestrator
from src.memory.manager import MemoryManager


async def run_audit():
    orchestrator = AdminOrchestrator()
    memory = MemoryManager()
    session_id = "surgical_audit_test"

    # Flush Redis session for isolation
    await memory.redis_client.delete(f"agent_state:{session_id}")
    await memory.redis_client.delete(f"message_store:{session_id}")

    print("\n--- TEST 1: Anti-Looping & Urgency Persistence ---")
    # Step 1: User introduces themselves with 10 months stay
    q1 = "Tôi sống ở Lyon được 11 tháng rồi. Muốn đổi bằng lái xe Việt Nam sang Pháp"
    print(f"User: {q1}")
    async for event in orchestrator.stream_query(q1, "vi", session_id):
        if event["type"] == "token":
            print(event["content"], end="", flush=True)
    print("\n")

    # Step 2: User says they don't have a translation
    q2 = "Tôi chưa có bản dịch công chứng"
    print(f"User: {q2}")
    result = ""
    async for event in orchestrator.stream_query(q2, "vi", session_id):
        if event["type"] == "token":
            result += event["content"]
            print(event["content"], end="", flush=True)
    print("\n")

    # EVALUATION
    if "KHẨN CẤP" in result or "11 tháng" in result or "hết hạn" in result:
        print("✅ URGENCY PERSISTED")
    else:
        print("❌ URGENCY LOST")

    if "dịch" in result.lower() and (
        "tìm" in result.lower()
        or "tòa" in result.lower()
        or "assermenté" in result.lower()
    ):
        print("✅ SOLUTION PROVIDED")
    else:
        print("❌ SOLUTION MISSING (OR REPETITION)")

    print("\n--- TEST 2: ANTS Priority ---")
    if "ants.gouv.fr" in result:
        print("✅ ANTS LINK FOUND")
    else:
        print("❌ ANTS LINK MISSING")


if __name__ == "__main__":
    asyncio.run(run_audit())

import os
import redis
from src.agents.orchestrator import AdminOrchestrator
from dotenv import load_dotenv

load_dotenv()

def test_memory():
    print("--- 🧠 Starting Memory Integration Test ---")
    orchestrator = AdminOrchestrator()
    session_id = "test_session_123"
    
    # Clear session history first for clean test
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis.from_url(redis_url)
    r.delete(f"message_store:{session_id}")
    print(f"Cleaned session: {session_id}")

    # 1. First turn: Introduce name and city
    print("\n[Turn 1] Query: My name is Anh and I come from Hanoi. How do I apply for a French passport?")
    res1 = orchestrator.handle_query("My name is Anh and I come from Hanoi. How do I apply for a French passport?", user_lang="en", session_id=session_id)
    print(f"Response (English):\n{res1}")

    # 2. Second turn: Ask about the name introduced in turn 1
    print("\n[Turn 2] Query: Do you remember my name? Answer in Vietnamese.")
    res2 = orchestrator.handle_query("Do you remember my name? Answer in Vietnamese.", user_lang="vi", session_id=session_id)
    print(f"Response (Vietnamese):\n{res2}")
    
    if "Anh" in res2:
        print("\n✅ SUCCESS: Memory works! The agent remembered the name.")
    else:
        print("\n❌ FAILURE: Memory issue. The agent forgot the name.")

    # 3. Third turn: Ask about the city introduced in turn 1
    print("\n[Turn 3] Query: Do you remember my city? Answer in Vietnamese.")
    res3 = orchestrator.handle_query("Do you remember my city? Answer in Vietnamese.", user_lang="vi", session_id=session_id)
    print(f"Response (Vietnamese):\n{res3}")
    
    if "Hà Nội" in res3 or "Hanoi" in res3:
        print("\n✅ SUCCESS: Memory works! The agent remembered the city.")
    else:
        print("\n❌ FAILURE: Memory issue. The agent forgot the city.")

if __name__ == "__main__":
    try:
        test_memory()
    except Exception as e:
        print(f"Error during memory test: {e}")
        print("Ensure Redis is running (e.g., docker-compose up redis).")

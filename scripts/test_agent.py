import os
import antigravity as ag
from src.agents.orchestrator import AdminOrchestrator
from dotenv import load_dotenv

load_dotenv()

def test_dry_run():
    print("--- 🧪 Starting Dry Run Test ---")
    orchestrator = AdminOrchestrator()
    
    # 1. Test standard query
    print("\n[Test 1] Query: Làm thế nào để đổi bằng lái xe ở Pháp? (Vietnamese)")
    try:
        # Mocking retriever result for this test to avoid needing Qdrant fully populated
        # In actual use, make sure Qdrant is running and ingested.
        res = orchestrator.handle_query("Làm thế nào để đổi bằng lái xe ở Pháp?", user_lang="vi")
        print(f"Response:\n{res}")
    except Exception as e:
        print(f"Error in Test 1: {e}")

    # 2. Test Guardrail (Topic Validation)
    print("\n[Test 2] Query: How to cook Pho? (Out of scope)")
    res_guard = orchestrator.handle_query("How to cook Pho?", user_lang="en")
    print(f"Response:\n{res_guard}")

    # 3. Test French query (Direct)
    print("\n[Test 3] Query: Comment renouveler mon passeport français ? (French)")
    res_fr = orchestrator.handle_query("Comment renouveler mon passeport français ?", user_lang="fr")
    print(f"Response:\n{res_fr}")

if __name__ == "__main__":
    test_dry_run()

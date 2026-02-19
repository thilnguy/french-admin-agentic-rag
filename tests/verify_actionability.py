import asyncio
from src.agents.orchestrator import AdminOrchestrator


# Mocking the interaction loop
async def run_conversation():
    orchestrator = AdminOrchestrator()
    session_id = "test_session_vietnam_license"

    # 1. User introduces themselves
    print("\n--- TURN 1: User Introduction ---")
    q1 = "Tôi là người Việt Nam, hiện đang sống tại Lyon. Tôi muốn hỏi về việc đổi bằng lái xe nước ngoài sang bằng Pháp"
    resp1 = await orchestrator.handle_query(q1, user_lang="vi", session_id=session_id)
    print(f"Agent: {resp1}")

    # 2. User answers 'Rồi' (or equivalent context confirmation if Agent asked)
    # Note: In the user report, the agent asked about the simulator.
    # Let's assume the previous turn worked somewhat and we are now testing the "Short Answer" trap.
    # But to be precise, I need to see what the agent *actually* asks in Turn 1.

    # 3. User says "Tôi đã cư trú tại Pháp được 10 tháng rồi"
    print("\n--- TURN 2: Durée de résidence ---")
    q2 = "Tôi đã cư trú tại Pháp được 10 tháng rồi"
    resp2 = await orchestrator.handle_query(q2, session_id=session_id)
    print(f"Agent: {resp2}")

    # 4. Agent SHOULD ask about long term plans. User says "Có" / "Rồi" / "Yes"
    print("\n--- TURN 3: Verification (The 'Yes' Trap) ---")
    q3 = "Rồi"  # Or "Có"
    resp3 = await orchestrator.handle_query(q3, session_id=session_id)
    print(f"Agent: {resp3}")

    # Check for failure keywords
    if "Désolé" in resp3 or "cannot treat" in resp3:
        print("\n❌ FAILURE DETECTED: Agent rejected the short answer 'Rồi'")
    else:
        print("\n✅ SUCCESS: Agent accepted 'Rồi'")

    # Check for language drift
    if "[GIVE]" in resp3 or "[DONNER]" in resp3:
        print(
            "\n❌ LANGUAGE ID FAILURE: Agent used English/French tags for Vietnamese query"
        )

    # --- PHASE 20 VERIFICATION ---
    print("\n--- PHASE 20: ACTIONABILITY CHECK ---")

    # Check for ANTS link
    if "ants.gouv.fr" in resp2 or "ants.gouv.fr" in resp3:
        print("✅ SUCCESS: Found ANTS link")
    else:
        print("❌ FAILURE: Missing ANTS link (Actionability)")

    # Check for Urgency Warning (Flexible matching for VN/FR/EN)
    urgency_keywords = [
        "URGENT",
        "KHẨN CẤP",
        "LƯU Ý QUAN TRỌNG",
        "ATTENTION",
        "DÉLAI LIMITE",
    ]
    # We expect warning in Turn 2 (Residency Duration) or Turn 3 (Confirmation)
    found_urgency = any(k in resp2.upper() for k in urgency_keywords) or any(
        k in resp3.upper() for k in urgency_keywords
    )

    if found_urgency:
        print("✅ SUCCESS: Found Urgency Warning")
    else:
        print("❌ FAILURE: Missing Urgency Warning for 10-month residency")

    # Check for Generic Closing Ban
    generic_closings = [
        "n'hésitez pas",
        "let me know",
        "hãy cho tôi biết",
        "cần thêm thông tin",
    ]
    found_generic = any(k in resp3.lower() for k in generic_closings)

    if found_generic:
        print("❌ FAILURE: Found banned generic closing ('Let me know...')")
    else:
        print("✅ SUCCESS: No generic closing found")


if __name__ == "__main__":
    import redis

    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        r.delete("agent_state:test_session_vietnam_license")
        print("Cleared Redis state.")
    except Exception:
        pass

    asyncio.run(run_conversation())

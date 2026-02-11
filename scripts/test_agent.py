import asyncio
from src.agents.orchestrator import AdminOrchestrator


async def main():
    agent = AdminOrchestrator()

    # Test 1: General question (RAG + Translation)
    query = "Làm thế nào để xin visa du học Pháp?"
    print(f"Query: {query}")
    response = await agent.handle_query(query, user_lang="vi")
    print(f"Response:\n{response}")

    # Test 2: Follow-up (Memory)
    query2 = "Hồ sơ cần những gì?"
    print(f"\nQuery 2: {query2}")
    response2 = await agent.handle_query(query2, user_lang="vi")
    print(f"Response 2:\n{response2}")


if __name__ == "__main__":
    asyncio.run(main())

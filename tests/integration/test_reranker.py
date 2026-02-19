import asyncio
from unittest.mock import MagicMock, patch
from src.agents.orchestrator import AdminOrchestrator
from src.agents.state import UserProfile


async def test_layer3_reranker_integration():
    print("Testing Layer 3: Context-Aware Reranker Integration...")

    # Mock Reranker
    with patch("skills.legal_retriever.main.get_reranker") as mock_get_reranker:
        mock_reranker_instance = MagicMock()
        mock_reranker_instance.rerank.return_value = [
            {"content": "Mock Doc", "source": "test", "score": 0.9}
        ]
        mock_get_reranker.return_value = mock_reranker_instance

        # Mock Retriever internal search (to avoid Qdrant/embedding calls if possible, but retrieve_legal_info calls them)
        # Actually better to patch retrieve_legal_info itself?
        # But we want to test that retrieve_legal_info calls reranker.
        # So we patch Qdrant/Embeddings inside retrieve_legal_info?
        # Let's patch 'skills.legal_retriever.main.QdrantClient' and 'skills.legal_retriever.main.HuggingFaceEmbeddings'

        with (
            patch("skills.legal_retriever.main.QdrantClient"),
            patch("skills.legal_retriever.main.HuggingFaceEmbeddings"),
        ):
            # We also need to mock the vector search result
            # But let's just run the orchestrator and see if reranker.rerank is called with profile.

            # Initialize
            _ = AdminOrchestrator()

            # Manually set a profile in state (simulating Layer 2)
            # We can't easily inject state into handle_query without mocking memory.
            # But handle_query loads state.

            # Let's just call retrieve_legal_info directly?
            # But we want to verify Orchestrator -> Retriever -> Reranker flow.

            # Let's use the Orchestrator but mock memory to return a state with profile.
            pass

    # Simplified Test: Call retrieve_legal_info directly
    print("Test 1: Calling retrieve_legal_info with profile")
    from skills.legal_retriever.main import retrieve_legal_info

    profile = UserProfile(location="Paris", nationality="Américaine")

    # Mock QdrantVectorStore to avoid initialization errors
    with (
        patch("skills.legal_retriever.main.QdrantVectorStore") as mock_vectorstore_cls,
        patch("skills.legal_retriever.main.HuggingFaceEmbeddings"),
    ):
        # Mock vector store instance
        mock_vs = mock_vectorstore_cls.return_value

        # Mock asimilarity_search (async)
        from langchain_core.documents import Document

        # Create a coroutine mock
        async def mock_asimilarity_search(*args, **kwargs):
            return [
                Document(page_content="Doc 1", metadata={"source": "url1"}),
                Document(page_content="Doc 2", metadata={"source": "url2"}),
            ]

        mock_vs.asimilarity_search.side_effect = mock_asimilarity_search

        with patch("skills.legal_retriever.main.get_reranker") as mock_get_reranker:
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = [
                {"content": "Doc 1", "source": "url1", "score": 0.9}
            ]
            mock_get_reranker.return_value = mock_reranker

            # Run
            await retrieve_legal_info("query", user_profile=profile)

            # Verify Reranker called with profile
            args, kwargs = mock_reranker.rerank.call_args
            called_profile = kwargs.get("user_profile")

            assert (
                called_profile == profile
            ), f"Reranker called with wrong profile: {called_profile}"
            print("SUCCESS: Reranker was called with UserProfile.")


if __name__ == "__main__":
    asyncio.run(test_layer3_reranker_integration())

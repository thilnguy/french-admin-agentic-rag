import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.orchestrator import AdminOrchestrator

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test that orchestrator initializes correctly."""
    with patch("src.agents.orchestrator.redis.Redis") as mock_redis, \
         patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"):
        orchestrator = AdminOrchestrator()
        assert orchestrator.llm is not None
        assert orchestrator.cache is not None

@pytest.mark.asyncio
async def test_handle_query_cache_hit():
    """Test that cache returns value if present."""
    with patch("src.agents.orchestrator.redis.Redis") as mock_redis_cls:
        # Mock Redis instance
        mock_redis = AsyncMock()
        mock_redis_cls.return_value = mock_redis
        
        # Setup cache hit
        mock_redis.get.return_value = "Cached Response"
        
        orchestrator = AdminOrchestrator()
        # Inject our mock instance (since __init__ creates a new one from the class mock)
        orchestrator.cache = mock_redis
        
        # Ensure DEBUG is False for cache to work
        with patch("src.config.settings.DEBUG", False), \
             patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"):
            orchestrator = AdminOrchestrator()
            # Inject our mock instance (since __init__ creates a new one from the class mock)
            orchestrator.cache = mock_redis
            
            response = await orchestrator.handle_query("test query", "fr")
            assert response == "Cached Response"
            mock_redis.get.assert_called_once()

@pytest.mark.asyncio
async def test_handle_query_flow():
    """Test the full flow with mocks."""
    with patch("src.agents.orchestrator.redis.Redis") as mock_redis_cls, \
         patch("src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock) as mock_retriever, \
         patch("src.agents.orchestrator.translate_admin_text", new_callable=AsyncMock) as mock_translator, \
         patch("src.shared.guardrails.guardrail_manager.validate_topic", new_callable=AsyncMock) as mock_validate, \
         patch("src.shared.guardrails.guardrail_manager.check_hallucination", new_callable=AsyncMock) as mock_hallucination, \
         patch("src.shared.guardrails.guardrail_manager.add_disclaimer", return_value="Final Answer + Disclaimer"), \
         patch("src.config.settings.OPENAI_API_KEY", "sk-test-key-mock"):
         
        # Setup Mocks
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None # Cache miss
        mock_redis_cls.return_value = mock_redis
        
        orchestrator = AdminOrchestrator()
        orchestrator.cache = mock_redis
        orchestrator.llm = MagicMock()
        orchestrator.llm.ainvoke = AsyncMock(return_value=MagicMock(content="Generated Answer"))
        
        mock_validate.return_value = (True, "")
        mock_retriever.return_value = [{"source": "test", "content": "context", "metadata": {}}]
        mock_hallucination.return_value = True
        
        # Execute
        response = await orchestrator.handle_query("Valid query", "fr")
        
        # Verify
        mock_validate.assert_called_once()
        mock_retriever.assert_called_once()
        orchestrator.llm.ainvoke.assert_called_once()
        assert response == "Final Answer + Disclaimer"

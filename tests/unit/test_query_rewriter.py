import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.preprocessor import QueryRewriter
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.mark.asyncio
async def test_rewrite_with_history():
    fake_llm = FakeListChatModel(responses=["Comment renouveler mon passeport à Le Cannet ?"])
    with patch("src.agents.preprocessor.get_llm", return_value=fake_llm):
        rewriter = QueryRewriter()

        history = [
            HumanMessage(content="J'habite à Le Cannet."),
            AIMessage(content="C'est noté."),
        ]
        query = "Comment le renouveler ici ?"

        result = await rewriter.rewrite(query, history)

        assert result == "Comment renouveler mon passeport à Le Cannet ?"


@pytest.mark.asyncio
async def test_rewrite_empty_history():
    rewriter = QueryRewriter()
    query = "Bonjour"

    result = await rewriter.rewrite(query, [])

    assert result == "Bonjour"


@pytest.mark.asyncio
async def test_rewrite_failure_fallback():
    fake_llm = FakeListChatModel(responses=["OK"])
    with patch.object(FakeListChatModel, "ainvoke", new_callable=AsyncMock, side_effect=Exception("API Error")):
        with patch("src.agents.preprocessor.get_llm", return_value=fake_llm):
            rewriter = QueryRewriter()

            history = [HumanMessage(content="Context")]
            query = "Test query"

            result = await rewriter.rewrite(query, history)

            # Should fallback to original query
            assert result == "Test query"

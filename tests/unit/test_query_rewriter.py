import pytest
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.preprocessor import QueryRewriter


@pytest.mark.asyncio
async def test_rewrite_with_history():
    rewriter = QueryRewriter()
    # Mock LLM chain to avoid API calls
    rewriter.chain = AsyncMock()
    rewriter.chain.ainvoke.return_value = (
        "Comment renouveler mon passeport à Le Cannet ?"
    )

    history = [
        HumanMessage(content="J'habite à Le Cannet."),
        AIMessage(content="C'est noté."),
    ]
    query = "Comment le renouveler ici ?"

    result = await rewriter.rewrite(query, history)

    assert result == "Comment renouveler mon passeport à Le Cannet ?"
    rewriter.chain.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_rewrite_empty_history():
    rewriter = QueryRewriter()
    query = "Bonjour"

    result = await rewriter.rewrite(query, [])

    assert result == "Bonjour"


@pytest.mark.asyncio
async def test_rewrite_failure_fallback():
    rewriter = QueryRewriter()
    rewriter.chain = AsyncMock()
    rewriter.chain.ainvoke.side_effect = Exception("API Error")

    history = [HumanMessage(content="Context")]
    query = "Test query"

    result = await rewriter.rewrite(query, history)

    # Should fallback to original query
    assert result == "Test query"

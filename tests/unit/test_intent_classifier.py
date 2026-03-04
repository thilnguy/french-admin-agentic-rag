import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.intent_classifier import IntentClassifier, Intent


@pytest.mark.asyncio
async def test_classify_success():
    with patch("src.agents.intent_classifier.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock the chain created by (prompt | llm)
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="LEGAL_INQUIRY"))
        
        # Patch __or__ so that `self.prompt | llm` returns our mock_chain
        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            classifier = IntentClassifier()
            intent = await classifier.classify("Some question")
            assert intent == Intent.LEGAL_INQUIRY


@pytest.mark.asyncio
async def test_classify_lowercase_handling():
    with patch("src.agents.intent_classifier.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="simple_qa"))
        
        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            classifier = IntentClassifier()
            intent = await classifier.classify("Some question")
            assert intent == Intent.SIMPLE_QA


@pytest.mark.asyncio
async def test_classify_unknown_enum():
    with patch("src.agents.intent_classifier.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="INVALID_INTENT"))
        
        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            classifier = IntentClassifier()
            intent = await classifier.classify("Some question")
            assert intent == Intent.UNKNOWN


@pytest.mark.asyncio
async def test_classify_exception_handling():
    with patch("src.agents.intent_classifier.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        with patch("langchain_core.prompts.ChatPromptTemplate.__or__", return_value=mock_chain):
            classifier = IntentClassifier()
            intent = await classifier.classify("Some question")
            assert intent == Intent.UNKNOWN

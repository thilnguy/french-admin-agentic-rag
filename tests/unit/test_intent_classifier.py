import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.intent_classifier import IntentClassifier, Intent


@pytest.mark.asyncio
async def test_classify_success():
    with patch("src.agents.intent_classifier.ChatOpenAI"):
        classifier = IntentClassifier()
        # Mock the chain
        classifier.chain = MagicMock()
        classifier.chain.ainvoke = AsyncMock(
            return_value=MagicMock(content="LEGAL_INQUIRY")
        )

        intent = await classifier.classify("Some question")
        assert intent == Intent.LEGAL_INQUIRY


@pytest.mark.asyncio
async def test_classify_lowercase_handling():
    with patch("src.agents.intent_classifier.ChatOpenAI"):
        classifier = IntentClassifier()
        classifier.chain = MagicMock()
        classifier.chain.ainvoke = AsyncMock(
            return_value=MagicMock(content="simple_qa")
        )

        # Code strips and upper()s the result
        intent = await classifier.classify("Some question")
        assert intent == Intent.SIMPLE_QA


@pytest.mark.asyncio
async def test_classify_unknown_enum():
    with patch("src.agents.intent_classifier.ChatOpenAI"):
        classifier = IntentClassifier()
        classifier.chain = MagicMock()
        # Return something not in Enum
        classifier.chain.ainvoke = AsyncMock(
            return_value=MagicMock(content="INVALID_INTENT")
        )

        intent = await classifier.classify("Some question")
        assert intent == Intent.UNKNOWN


@pytest.mark.asyncio
async def test_classify_exception_handling():
    with patch("src.agents.intent_classifier.ChatOpenAI"):
        classifier = IntentClassifier()
        classifier.chain = MagicMock()
        classifier.chain.ainvoke = AsyncMock(side_effect=Exception("API Error"))

        intent = await classifier.classify("Some question")
        assert intent == Intent.UNKNOWN

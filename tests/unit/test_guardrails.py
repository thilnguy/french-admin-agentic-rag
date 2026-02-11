import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_validate_topic_admin_question_approved():
    """Administrative question should be APPROVED."""
    with patch("src.shared.guardrails.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        # Mock the chain invoke
        with patch.object(gm, "llm") as mock_llm_instance:
            mock_llm_instance.__or__ = MagicMock()
            with patch("src.shared.guardrails.ChatPromptTemplate") as mock_prompt:
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value="APPROVED")
                mock_prompt.from_messages.return_value.__or__ = MagicMock(
                    return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
                )

                is_valid, reason = await gm.validate_topic(
                    "Comment obtenir un passeport ?"
                )
                assert is_valid is True
                assert reason == ""


@pytest.mark.asyncio
async def test_validate_topic_unrelated_rejected():
    """Unrelated question should be REJECTED."""
    with patch("src.shared.guardrails.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()

        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        with patch("src.shared.guardrails.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(
                return_value="REJECTED: Question non liée à l'administration française"
            )
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            is_valid, reason = await gm.validate_topic("How to cook pasta?")
            assert is_valid is False
            assert "non liée" in reason


@pytest.mark.asyncio
async def test_validate_topic_followup_with_history_approved():
    """Follow-up question with admin history should be APPROVED."""
    with patch("src.shared.guardrails.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()

        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        mock_history = [
            MagicMock(type="human", content="Comment renouveler mon titre de séjour ?"),
            MagicMock(
                type="ai", content="Pour renouveler, rendez-vous à la préfecture..."
            ),
        ]

        with patch("src.shared.guardrails.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="APPROVED")
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            is_valid, reason = await gm.validate_topic("Why?", history=mock_history)
            assert is_valid is True


@pytest.mark.asyncio
async def test_check_hallucination_safe():
    """Answer grounded in context should return True (SAFE)."""
    with patch("src.shared.guardrails.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()

        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        with patch("src.shared.guardrails.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="SAFE")
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            result = await gm.check_hallucination(
                context="Le passeport coûte 86€.",
                answer="Le coût du passeport est de 86€.",
                query="Combien coûte un passeport ?",
            )
            assert result is True


@pytest.mark.asyncio
async def test_check_hallucination_detected():
    """Fabricated answer should return False (HALLUCINATION)."""
    with patch("src.shared.guardrails.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()

        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        with patch("src.shared.guardrails.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="HALLUCINATION")
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            result = await gm.check_hallucination(
                context="Le passeport coûte 86€.",
                answer="Le passeport est gratuit pour tous les résidents.",
                query="Combien coûte un passeport ?",
            )
            assert result is False


def test_add_disclaimer_french():
    """French disclaimer should be fully in French."""
    with patch("src.shared.guardrails.ChatOpenAI"):
        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        result = gm.add_disclaimer("Test answer", "fr")
        assert "service-public.fr" in result
        assert "informations sont données à titre indicatif" in result
        assert "Test answer" in result


def test_add_disclaimer_english():
    """English disclaimer should be in English."""
    with patch("src.shared.guardrails.ChatOpenAI"):
        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        result = gm.add_disclaimer("Test answer", "en")
        assert "guidance only" in result


def test_add_disclaimer_unknown_language_defaults_to_french():
    """Unknown language should fall back to French disclaimer."""
    with patch("src.shared.guardrails.ChatOpenAI"):
        from src.shared.guardrails import GuardrailManager

        gm = GuardrailManager()

        result = gm.add_disclaimer("Test answer", "de")
        assert "informations sont données à titre indicatif" in result

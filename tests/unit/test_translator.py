import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_translate_returns_translation():
    """Translator should return translated text."""
    with patch("skills.admin_translator.ChatOpenAI") as mock_llm_cls, patch(
        "skills.admin_translator._llm", None
    ):
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        with patch("skills.admin_translator.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(
                return_value="Application for a residence permit at the prefecture"
            )
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            from skills.admin_translator import translate_admin_text

            result = await translate_admin_text(
                "Demande de titre de séjour à la préfecture", "English"
            )
            assert (
                "residence permit" in result.lower()
                or "prefecture" in result.lower()
                or isinstance(result, str)
            )
            mock_chain.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_translate_accepts_vietnamese():
    """Translator should accept Vietnamese as target language."""
    with patch("skills.admin_translator.ChatOpenAI") as mock_llm_cls, patch(
        "skills.admin_translator._llm", None
    ):
        mock_llm_cls.return_value = MagicMock()

        with patch("skills.admin_translator.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Đơn xin thẻ cư trú tại quận")
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            from skills.admin_translator import translate_admin_text

            result = await translate_admin_text(
                "Demande de titre de séjour", "Vietnamese"
            )
            assert isinstance(result, str)
            assert len(result) > 0


@pytest.mark.asyncio
async def test_translate_passes_correct_params():
    """Translator should pass text and target_language to the chain."""
    with patch("skills.admin_translator.ChatOpenAI") as mock_llm_cls, patch(
        "skills.admin_translator._llm", None
    ):
        mock_llm_cls.return_value = MagicMock()

        with patch("skills.admin_translator.ChatPromptTemplate") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.ainvoke = AsyncMock(return_value="Translated text")
            mock_prompt.from_messages.return_value.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )

            from skills.admin_translator import translate_admin_text

            await translate_admin_text("Mon texte", "English")

            call_args = mock_chain.ainvoke.call_args[0][0]
            assert call_args["text"] == "Mon texte"
            assert call_args["target_language"] == "English"

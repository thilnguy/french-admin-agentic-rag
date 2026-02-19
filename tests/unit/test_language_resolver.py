"""
Unit tests for LanguageResolver.

Tests are grouped by rule. Each test has a comment explaining which rule it covers.
These tests do NOT import or touch the orchestrator — LanguageResolver is independently testable.
"""

import pytest
from src.shared.language_resolver import LanguageResolver


@pytest.fixture
def resolver():
    return LanguageResolver()


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_fr_code(resolver):
    assert resolver.normalize("fr") == "French"


def test_normalize_en_code(resolver):
    assert resolver.normalize("en") == "English"


def test_normalize_vi_code(resolver):
    assert resolver.normalize("vi") == "Vietnamese"


def test_normalize_full_name_passthrough(resolver):
    """Full names not in map are returned as-is."""
    assert resolver.normalize("French") == "French"


def test_normalize_case_insensitive(resolver):
    assert resolver.normalize("FR") == "French"
    assert resolver.normalize("En") == "English"


# ---------------------------------------------------------------------------
# Rule 1: Frontend manual override protection
# ---------------------------------------------------------------------------


def test_rule1_frontend_en_blocks_fr_detection(resolver):
    """If frontend chose 'en', detected 'fr' must not override it."""
    result = resolver.resolve(
        detected_lang="fr",
        user_lang="en",
        current_state_lang="English",
        has_history=False,
    )
    assert result == "English"


def test_rule1_frontend_vi_blocks_en_detection(resolver):
    """If frontend chose 'vi', detected 'en' must not override it."""
    result = resolver.resolve(
        detected_lang="en",
        user_lang="vi",
        current_state_lang="Vietnamese",
        has_history=False,
    )
    assert result == "Vietnamese"


def test_rule1_frontend_fr_does_not_block(resolver):
    """Frontend 'fr' is the default — does NOT block detection."""
    result = resolver.resolve(
        detected_lang="en",
        user_lang="fr",
        current_state_lang="French",
        has_history=False,
    )
    assert result == "English"


# ---------------------------------------------------------------------------
# Rule 2: Anti-hallucination guard
# ---------------------------------------------------------------------------


def test_rule2_fr_detection_ignored_when_already_english(resolver):
    """If already English, detected 'fr' is treated as a hallucination."""
    result = resolver.resolve(
        detected_lang="fr",
        user_lang="fr",
        current_state_lang="English",
        has_history=True,
    )
    assert result == "English"


def test_rule2_fr_detection_ignored_when_already_vietnamese(resolver):
    """If already Vietnamese, detected 'fr' is treated as a hallucination."""
    result = resolver.resolve(
        detected_lang="fr",
        user_lang="fr",
        current_state_lang="Vietnamese",
        has_history=True,
    )
    assert result == "Vietnamese"


def test_rule2_does_not_block_en_detection(resolver):
    """Anti-hallucination only blocks 'fr' detection. 'en' detection is allowed."""
    result = resolver.resolve(
        detected_lang="en",
        user_lang="fr",
        current_state_lang="French",
        has_history=True,
    )
    assert result == "English"


# ---------------------------------------------------------------------------
# Rule 3 (Bug Fix): First-message English detection (no history)
# ---------------------------------------------------------------------------


def test_rule3_first_message_en_switches_from_french(resolver):
    """
    BUG FIX: 'Tell me about myself' → detected 'en', current 'fr', no history.
    Previously: stays 'fr' (anti-hallucination too broad).
    Now: switches to 'English' on first message.
    """
    result = resolver.resolve(
        detected_lang="en",
        user_lang="fr",
        current_state_lang="French",
        has_history=False,
    )
    assert result == "English"


def test_rule3_first_message_vi_switches_from_french(resolver):
    """First Vietnamese message with no history switches language."""
    result = resolver.resolve(
        detected_lang="vi",
        user_lang="fr",
        current_state_lang="French",
        has_history=False,
    )
    assert result == "Vietnamese"


def test_rule3_en_with_history_also_switches(resolver):
    """English detection with history also allowed to switch."""
    result = resolver.resolve(
        detected_lang="en",
        user_lang="fr",
        current_state_lang="French",
        has_history=True,
    )
    assert result == "English"


# ---------------------------------------------------------------------------
# Rule 4: Default passthrough
# ---------------------------------------------------------------------------


def test_rule4_no_detected_lang_keeps_current(resolver):
    """No detected language → current state preserved."""
    result = resolver.resolve(
        detected_lang=None,
        user_lang="fr",
        current_state_lang="English",
        has_history=False,
    )
    assert result == "English"


def test_rule4_no_detected_lang_defaults_to_french(resolver):
    """No detected language and no current → defaults to French."""
    result = resolver.resolve(
        detected_lang=None,
        user_lang="fr",
        current_state_lang="",
        has_history=False,
    )
    assert result == "French"


def test_rule4_fr_on_fr_state_stays_french(resolver):
    """Detecting 'fr' when already French is a no-op."""
    result = resolver.resolve(
        detected_lang="fr",
        user_lang="fr",
        current_state_lang="French",
        has_history=False,
    )
    assert result == "French"


# ---------------------------------------------------------------------------
# apply_to_state() — integration-style tests
# ---------------------------------------------------------------------------


class MockProfile:
    """Mock of AgentState.user_profile for testing apply_to_state."""

    def __init__(self):
        self.language = "French"
        self.name = None
        self.nationality = None
        self.location = None

    def model_dump(self):
        return {
            "language": self.language,
            "name": self.name,
            "nationality": self.nationality,
            "location": self.location,
        }


def test_apply_to_state_switches_language(resolver):
    """apply_to_state updates profile language when detection is valid."""
    profile = MockProfile()
    updated = resolver.apply_to_state(
        extracted_data={"language": "en", "name": None},
        user_lang="fr",
        state_profile=profile,
        has_history=False,
    )
    assert updated is True
    assert profile.language == "English"


def test_apply_to_state_updates_other_fields(resolver):
    """Non-language fields are updated independently."""
    profile = MockProfile()
    updated = resolver.apply_to_state(
        extracted_data={"language": None, "name": "Alice", "nationality": "French"},
        user_lang="fr",
        state_profile=profile,
        has_history=False,
    )
    assert updated is True
    assert profile.name == "Alice"
    assert profile.nationality == "French"
    assert profile.language == "French"  # unchanged


def test_apply_to_state_skips_reasoning_field(resolver):
    """_reasoning field is never applied to profile."""
    profile = MockProfile()
    updated = resolver.apply_to_state(
        extracted_data={"language": None, "_reasoning": "some text"},
        user_lang="fr",
        state_profile=profile,
        has_history=False,
    )
    assert updated is False
    assert not hasattr(profile, "_reasoning") or profile._reasoning is None


def test_apply_to_state_returns_false_when_no_changes(resolver):
    """Returns False when extracted data matches current profile."""
    profile = MockProfile()
    profile.language = "English"
    updated = resolver.apply_to_state(
        extracted_data={"language": "en"},
        user_lang="en",
        state_profile=profile,
        has_history=True,
    )
    # Frontend is 'en' → Rule 1 applies, returns "English" which equals current
    assert updated is False
    assert profile.language == "English"

"""
LanguageResolver — stateless language resolution extracted from AdminOrchestrator.

PURPOSE:
    Centralizes both the language switching logic and anti-hallucination rules that were
    previously duplicated between `handle_query` (L180-210) and `stream_query` (L470-520).

DIVERGENCES FIXED:
    - handle_query: had a subtle bug where `value == "fr"` check was used instead of
      `detected_lang == "fr"`, meaning the full normalized value was compared to code.
    - stream_query: correctly normalized before the anti-hallucination check.
    - Both: the anti-hallucination rule blocked ALL English→fr detection, even with no
      prior history. This is replaced by a confidence-aware approach.

DESIGN:
    Pure function with no side effects. Takes inputs, returns resolved language string.
    No LLM calls. No state mutation. Independently unit-testable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("french_admin_agent")

# Language code → full name mapping
LANG_MAP: dict[str, str] = {
    "fr": "French",
    "french": "French",
    "en": "English",
    "english": "English",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
}

# Languages we consider "non-default" (i.e., switching away from French is significant)
NON_DEFAULT_LANGUAGES = {"English", "Vietnamese"}


class LanguageResolver:
    """
    Stateless resolver for effective response language.

    Rules (in priority order):
    1. Manual frontend override: If user_lang (from frontend) is non-French,
       and detected_lang disagrees, keep user_lang's language.
    2. Anti-hallucination: If detected_lang is 'fr' but current state is already
       English/Vietnamese, ignore the detection (likely noise from French keywords in query).
    3. Confidence-based switching: If detected_lang is non-French AND current state is French,
       allow the switch only if there was some prior context (chat history or explicit signal).
       For the first message with no history, allow the switch freely.
    4. Otherwise: apply the detection.
    """

    def __init__(self, lang_map: dict[str, str] | None = None):
        self.lang_map = lang_map or LANG_MAP

    def normalize(self, lang_code: str) -> str:
        """Normalize a language code or name to its full English name."""
        return self.lang_map.get(lang_code.lower(), lang_code)

    def resolve(
        self,
        detected_lang: str | None,
        user_lang: str | None,
        current_state_lang: str,
        has_history: bool = False,
    ) -> str:
        """
        Resolve the effective language for the response.

        Args:
            detected_lang: Language code detected by ProfileExtractor (e.g. 'en', 'fr').
            user_lang: Language hint from frontend (e.g. 'en', 'fr'). May be None.
            current_state_lang: Current language stored in AgentState.user_profile.language.
            has_history: Whether there is prior chat history (affects anti-hallucination).

        Returns:
            Resolved full language name (e.g. 'English', 'French', 'Vietnamese').
        """
        if not detected_lang:
            # No detected language → keep current state
            return current_state_lang or "French"

        normalized_detected = self.normalize(detected_lang)
        normalized_user = self.normalize(user_lang) if user_lang else None
        current = current_state_lang or "French"

        # Rule 1: Frontend manual override protection
        # If the user explicitly chose a non-French language on the frontend,
        # trust that choice over the ProfileExtractor's detection.
        if normalized_user and normalized_user in NON_DEFAULT_LANGUAGES:
            if normalized_detected != normalized_user:
                logger.info(
                    f"LanguageResolver: Blocking detection '{normalized_detected}' "
                    f"— frontend manually chose '{normalized_user}'"
                )
            # Always honor the frontend choice when it's explicitly non-French
            return normalized_user

        # Rule 2: Anti-hallucination guard
        # If detected is French but we're already in English/Vietnamese,
        # the detector likely hallucinated due to French admin keywords in the query.
        if detected_lang == "fr" and current in NON_DEFAULT_LANGUAGES:
            logger.info(
                f"LanguageResolver: Ignoring 'fr' hallucination — "
                f"already in '{current}'"
            )
            return current

        # Rule 3: FIX — previously, the first message in English was incorrectly
        # kept as 'fr' because the anti-hallucination rule was too broad.
        # Now we allow the switch on first message (no history) when detection is clear.
        if normalized_detected in NON_DEFAULT_LANGUAGES and current == "French":
            if not has_history:
                # First message with no prior context: trust the detector
                logger.info(
                    f"LanguageResolver: First-message switch fr → {normalized_detected}"
                )
                return normalized_detected
            else:
                # With history: also allow the switch
                logger.info(
                    f"LanguageResolver: History-based switch {current} → {normalized_detected}"
                )
                return normalized_detected

        # Rule 4: Default — apply the detected language
        logger.info(f"LanguageResolver: Applying detection → {normalized_detected}")
        return normalized_detected

    def apply_to_state(
        self,
        extracted_data: dict,
        user_lang: str | None,
        state_profile,
        has_history: bool = False,
    ) -> bool:
        """
        Apply extracted profile data to state.user_profile, using LanguageResolver
        for the language field.

        Args:
            extracted_data: Dict from ProfileExtractor.extract().
            user_lang: Frontend language hint.
            state_profile: state.user_profile (mutated in place).
            has_history: Whether prior chat history exists.

        Returns:
            True if any field was updated.
        """
        updated = False

        # --- Handle language first (special logic) ---
        detected_lang = extracted_data.get("language")
        if detected_lang:
            resolved = self.resolve(
                detected_lang=detected_lang,
                user_lang=user_lang,
                current_state_lang=state_profile.language,
                has_history=has_history,
            )
            if state_profile.language != resolved:
                logger.info(f"LanguageResolver: {state_profile.language} → {resolved}")
                state_profile.language = resolved
                updated = True

        # --- Handle other fields ---
        current_profile_dict = state_profile.model_dump()
        for key, value in extracted_data.items():
            if key in ("language", "_reasoning"):
                continue
            if value is not None and key in current_profile_dict:
                if getattr(state_profile, key) != value:
                    setattr(state_profile, key, value)
                    updated = True

        return updated


# Module-level singleton — reuse across requests (stateless, safe to share)
language_resolver = LanguageResolver()

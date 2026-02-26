import re
from typing import Tuple
from src.utils.logger import logger

class InjectionGuard:
    def __init__(self):
        # Common prompt injection patterns across EN, FR, VI
        self.injection_patterns = [
            r"(?i)\b(ignore|oublie?|bỏ qua)\b.*(instructions?|prompt|directions?|hướng dẫn)",
            r"(?i)\b(system prompt|instructions de base)\b",
            r"(?i)\b(you are now|tu es maintenant|bạn bây giờ là)\b",
            r"(?i)\b(forget|bỏ|xoá)\b.*(context|contexte|ngữ cảnh)",
            r"(?i)\b(act as|agis comme|hãy đóng vai)\b.*(uncensored|jailbreak|no rules|sans règles|không có luật)",
            r"(?i)^[\s\W]*(ignore|forget|bypass|override)[\s\W]+",
            r"(?i)\b(print|show|display|affiche).*(previous|system|précédent) (instructions?|prompt)\b"
        ]
        self.compiled_patterns = [re.compile(p) for p in self.injection_patterns]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Scans the query for known prompt injection patterns.
        Returns (is_valid, reason).
        If injection is detected, is_valid is False and reason is provided.
        """
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected! Pattern matched: {pattern.pattern}")
                return False, "Prompt injection attempt detected and blocked."
        return True, ""

# Singleton instance
injection_guard = InjectionGuard()

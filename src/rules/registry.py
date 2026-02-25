"""
Topic Registry — Loads and serves topic-specific rules from YAML.

Usage:
    from src.rules.registry import topic_registry
    
    topic = topic_registry.detect_topic(query, intent)
    rules = topic_registry.get_rules(topic)
    prompt_fragment = topic_registry.build_prompt_fragment(topic, user_profile)
"""

import os
import yaml
from typing import Optional, Dict, List, Any
from src.utils.logger import logger


class TopicRules:
    """Represents the rules for a single topic."""
    
    def __init__(self, key: str, data: dict):
        self.key = key
        self.display_name = data.get("display_name", key)
        self.description = data.get("description", "")
        self.default_step = data.get("default_step", "CLARIFICATION")
        self.mandatory_variables = data.get("mandatory_variables", [])
        self.conditional_variables = data.get("conditional_variables", [])
        self.exemplar = data.get("exemplar", {})
        self.guardrail_keywords = data.get("guardrail_keywords", [])
        self.force_retrieval_patterns = data.get("force_retrieval_patterns", [])
    
    def get_missing_variables(self, user_profile: dict) -> List[dict]:
        """Returns mandatory variables not yet present in user profile."""
        missing = []
        for var in self.mandatory_variables:
            var_name = var["name"]
            profile_value = user_profile.get(var_name)
            if not profile_value or profile_value == "None":
                missing.append(var)
        return missing
    
    def get_applicable_conditionals(self, query: str) -> List[dict]:
        """Returns conditional variables relevant to the current query."""
        applicable = []
        for var in self.conditional_variables:
            trigger = var.get("when", "").lower()
            if trigger and trigger in query.lower():
                applicable.append(var)
        return applicable
    
    def format_variable_list(self, variables: List[dict]) -> str:
        """Formats variables into a prompt-friendly string."""
        if not variables:
            return "No specific variables needed."
        lines = []
        for var in variables:
            lines.append(f"- {var['name']}: {var['why']}")
        return "\n".join(lines)
    
    def format_exemplar(self) -> str:
        """Formats the exemplar into a prompt-friendly string."""
        if not self.exemplar:
            return ""
        return f"""EXAMPLE for this topic:
Input: {self.exemplar.get('input', '')}
Expected output:
{self.exemplar.get('output', '')}"""


class TopicRegistry:
    """Central registry for all topic-specific rules."""
    
    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "topic_registry.yaml"
            )
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        
        self.topics: Dict[str, TopicRules] = {}
        for key, data in raw.get("topics", {}).items():
            self.topics[key] = TopicRules(key, data)
        
        self.global_rules = raw.get("global_rules", {})
        
        # Build keyword index for fast topic detection
        self._keyword_index: Dict[str, str] = {}
        for key, topic in self.topics.items():
            for kw in topic.guardrail_keywords:
                self._keyword_index[kw.lower()] = key
        
        logger.info(f"TopicRegistry loaded: {len(self.topics)} topics, {len(self._keyword_index)} keywords")
    
    def detect_topic(self, query: str, intent: str = None) -> str:
        """
        Detects the most likely topic for a query using keyword matching.
        Falls back to 'daily_life' if no match found.
        """
        query_lower = query.lower()
        
        # Count keyword hits per topic
        scores: Dict[str, int] = {}
        for kw, topic_key in self._keyword_index.items():
            if kw in query_lower:
                scores[topic_key] = scores.get(topic_key, 0) + 1
        
        if scores:
            best_topic = max(scores, key=scores.get)
            logger.debug(f"TopicDetector: '{query[:50]}...' → {best_topic} (score: {scores[best_topic]})")
            return best_topic
        
        # Fallback: use intent to guess
        if intent:
            intent_str = str(intent).upper()
            if "LEGAL" in intent_str:
                return "identity"
            if "FORM" in intent_str:
                return "immigration"
        
        logger.debug(f"TopicDetector: no keyword match, defaulting to 'daily_life'")
        return "daily_life"
    
    def get_rules(self, topic_key: str) -> Optional[TopicRules]:
        """Get rules for a specific topic."""
        return self.topics.get(topic_key)
    
    def build_prompt_fragment(self, topic_key: str, user_profile: dict = None, query: str = "") -> str:
        """
        Builds a focused prompt fragment with ONLY the relevant topic's rules.
        This replaces the massive inline rule blocks in the current prompts.
        """
        rules = self.get_rules(topic_key)
        if not rules:
            return ""
        
        user_profile = user_profile or {}
        
        # Get missing variables (mandatory + applicable conditionals)
        missing = rules.get_missing_variables(user_profile)
        conditionals = rules.get_applicable_conditionals(query)
        all_vars = missing + conditionals
        
        fragment = f"""
TOPIC: {rules.display_name}

VARIABLES YOU MUST ASK FOR (if not already known):
{rules.format_variable_list(all_vars) if all_vars else "All key variables are already known. Provide a direct answer."}

{rules.format_exemplar()}
"""
        return fragment.strip()
    
    def build_global_rules_fragment(self) -> str:
        """Builds the global rules section for any prompt."""
        lines = []
        for category, rules_list in self.global_rules.items():
            lines.append(f"\n{category.upper().replace('_', ' ')}:")
            for rule in rules_list:
                lines.append(f"- {rule}")
        return "\n".join(lines)


# Singleton
topic_registry = TopicRegistry()

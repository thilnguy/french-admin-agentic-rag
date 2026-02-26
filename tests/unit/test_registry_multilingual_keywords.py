"""
Temporary unittest for multilingual keyword dict refactor in topic_registry.

Tests:
  1. Backward compat — existing flat list format still works
  2. New dict format — keywords are correctly flattened from {fr:[], en:[], vi:[]}
  3. Topic detection works for queries in all 3 languages
  4. Mixed topics: one topic uses flat list, another uses dict format

Run with:
    pytest tests/unit/test_registry_multilingual_keywords.py -v
"""
import yaml
import textwrap
import pytest
from unittest.mock import patch, mock_open

# ── helper: build a minimal YAML string ────────────────────────────────────
FLAT_YAML = textwrap.dedent("""\
    persona: "Marianne AI"
    global_rules: {}
    topics:
      labor:
        display_name: "Work & Labor"
        description: "Labor rights"
        default_step: CLARIFICATION
        mandatory_variables: []
        conditional_variables: []
        guardrail_keywords:
          - "employeur"
          - "strike"
          - "tiền lương"
""")

DICT_YAML = textwrap.dedent("""\
    persona: "Marianne AI"
    global_rules: {}
    topics:
      labor:
        display_name: "Work & Labor"
        description: "Labor rights"
        default_step: CLARIFICATION
        mandatory_variables: []
        conditional_variables: []
        guardrail_keywords:
          fr: ["employeur", "heures supplémentaires", "grève"]
          en: ["strike", "unpaid wages"]
          vi: ["tiền lương", "hợp đồng lao động", "làm việc"]
""")

MIXED_YAML = textwrap.dedent("""\
    persona: "Marianne AI"
    global_rules: {}
    topics:
      labor:
        display_name: "Work & Labor"
        description: "Labor rights"
        default_step: CLARIFICATION
        mandatory_variables: []
        conditional_variables: []
        guardrail_keywords:
          fr: ["employeur", "grève"]
          en: ["strike"]
          vi: ["tiền lương"]
      taxes:
        display_name: "Taxes"
        description: "Tax management"
        default_step: CLARIFICATION
        mandatory_variables: []
        conditional_variables: []
        guardrail_keywords:
          - "impôts"
          - "tax"
          - "thuế"
""")


# ── patch TopicRegistry to load from string instead of file ────────────────
def make_registry(yaml_str: str):
    """Creates a TopicRegistry from a raw YAML string (no file I/O)."""
    from src.rules.registry import TopicRegistry
    raw = yaml.safe_load(yaml_str)
    registry = TopicRegistry.__new__(TopicRegistry)
    registry.topics = {}
    registry.global_rules = raw.get("global_rules", {})
    registry.persona = raw.get("persona", "")
    registry._keyword_index = {}

    from src.rules.registry import TopicRules
    for key, data in raw.get("topics", {}).items():
        registry.topics[key] = TopicRules(key, data)

    for key, topic in registry.topics.items():
        for kw in topic.guardrail_keywords:
            registry._keyword_index[kw.lower()] = key

    return registry


# ════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ════════════════════════════════════════════════════════════════

class TestFlatListBackwardCompat:
    """Existing flat list format must still work identically."""

    def test_keywords_loaded_as_list(self):
        registry = make_registry(FLAT_YAML)
        labor = registry.topics["labor"]
        assert isinstance(labor.guardrail_keywords, list)

    def test_all_three_keywords_present(self):
        registry = make_registry(FLAT_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert "employeur" in kwds
        assert "strike" in kwds
        assert "tiền lương" in kwds

    def test_detect_topic_french(self):
        registry = make_registry(FLAT_YAML)
        assert registry.detect_topic("Mon employeur ne me paie pas") == "labor"

    def test_detect_topic_english(self):
        registry = make_registry(FLAT_YAML)
        assert registry.detect_topic("I am joining a strike tomorrow") == "labor"

    def test_detect_topic_vietnamese(self):
        registry = make_registry(FLAT_YAML)
        assert registry.detect_topic("tiền lương của tôi chưa được trả") == "labor"


class TestDictFormatNewBehavior:
    """New dict format: {fr: [], en: [], vi: []} must be flattened correctly."""

    def test_keywords_flattened_to_list(self):
        registry = make_registry(DICT_YAML)
        labor = registry.topics["labor"]
        assert isinstance(labor.guardrail_keywords, list), \
            "guardrail_keywords should be flattened to a list even when input is a dict"

    def test_all_fr_keywords_present(self):
        registry = make_registry(DICT_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert "employeur" in kwds
        assert "heures supplémentaires" in kwds
        assert "grève" in kwds

    def test_all_en_keywords_present(self):
        registry = make_registry(DICT_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert "strike" in kwds
        assert "unpaid wages" in kwds

    def test_all_vi_keywords_present(self):
        registry = make_registry(DICT_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert "tiền lương" in kwds
        assert "hợp đồng lao động" in kwds
        assert "làm việc" in kwds

    def test_total_keyword_count(self):
        registry = make_registry(DICT_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert len(kwds) == 8, f"Expected 8 keywords (3 FR + 2 EN + 3 VI), got {len(kwds)}: {kwds}"

    def test_detect_topic_french(self):
        registry = make_registry(DICT_YAML)
        assert registry.detect_topic("Mon employeur paye pas les heures") == "labor"

    def test_detect_topic_english(self):
        registry = make_registry(DICT_YAML)
        assert registry.detect_topic("Can I get compensation for unpaid wages?") == "labor"

    def test_detect_topic_vietnamese(self):
        registry = make_registry(DICT_YAML)
        assert registry.detect_topic("Tôi đang làm việc không có hợp đồng") == "labor"

    def test_keyword_index_built_correctly(self):
        registry = make_registry(DICT_YAML)
        assert registry._keyword_index.get("grève") == "labor"
        assert registry._keyword_index.get("strike") == "labor"
        assert registry._keyword_index.get("tiền lương") == "labor"


class TestMixedFormats:
    """One topic uses new dict format, another uses legacy flat list — both work."""

    def test_both_topics_loaded(self):
        registry = make_registry(MIXED_YAML)
        assert "labor" in registry.topics
        assert "taxes" in registry.topics

    def test_labor_dict_format_flattened(self):
        registry = make_registry(MIXED_YAML)
        kwds = registry.topics["labor"].guardrail_keywords
        assert "employeur" in kwds and "strike" in kwds and "tiền lương" in kwds

    def test_taxes_flat_format_preserved(self):
        registry = make_registry(MIXED_YAML)
        kwds = registry.topics["taxes"].guardrail_keywords
        assert "impôts" in kwds and "tax" in kwds and "thuế" in kwds

    def test_detect_labor_vi(self):
        registry = make_registry(MIXED_YAML)
        assert registry.detect_topic("tiền lương không được trả") == "labor"

    def test_detect_taxes_vi(self):
        registry = make_registry(MIXED_YAML)
        assert registry.detect_topic("thuế năm nay tôi phải nộp bao nhiêu") == "taxes"

    def test_detect_taxes_fr(self):
        registry = make_registry(MIXED_YAML)
        assert registry.detect_topic("Je dois payer mes impôts cette année") == "taxes"


class TestTopicRulesKeywordNormalization:
    """TopicRules itself should expose a flat list regardless of input format."""

    def test_flat_input_returns_list(self):
        from src.rules.registry import TopicRules
        data = {"guardrail_keywords": ["a", "b", "c"]}
        rules = TopicRules("test", data)
        assert rules.guardrail_keywords == ["a", "b", "c"]

    def test_dict_input_returns_flat_list(self):
        from src.rules.registry import TopicRules
        data = {"guardrail_keywords": {"fr": ["a", "b"], "en": ["c"], "vi": ["d", "e"]}}
        rules = TopicRules("test", data)
        assert isinstance(rules.guardrail_keywords, list)
        assert set(rules.guardrail_keywords) == {"a", "b", "c", "d", "e"}

    def test_empty_keywords_returns_empty_list(self):
        from src.rules.registry import TopicRules
        rules = TopicRules("test", {})
        assert rules.guardrail_keywords == []

    def test_dict_with_empty_lang_list(self):
        from src.rules.registry import TopicRules
        data = {"guardrail_keywords": {"fr": ["a"], "en": [], "vi": []}}
        rules = TopicRules("test", data)
        assert rules.guardrail_keywords == ["a"]

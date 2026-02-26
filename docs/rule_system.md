# Rule System: Data-Driven Topic Registry

## Overview

The **Topic Registry** is the single source of truth for all agent behavior rules. Instead of embedding topic-specific logic deep inside Python prompts (which causes duplication and fragility), all rules are declared in `src/rules/topic_registry.yaml` and injected dynamically at runtime.

This was implemented as part of **Phase 1 of the Rule System Optimization** (see [architecture_evolution.md](architecture_evolution.md)).

---

## Architecture

```
src/rules/
├── topic_registry.yaml   ← Single source of truth for all rules
└── registry.py           ← TopicRegistry class (loads, indexes, injects)
```

The `TopicRegistry` class is a singleton initialized at startup. It provides:
- `detect_topic(query, intent)` — keyword-based topic classification.
- `build_prompt_fragment(topic, user_profile, query)` — injects only the *relevant* rules for the current topic into the prompt (60-70% shorter prompts than the monolith approach).
- `build_global_rules_fragment()` — injects universal rules applicable to all topics.

---

## `topic_registry.yaml` Format

Each topic is a key under `topics:`. Here is the full structure for a single topic:

```yaml
topics:
  immigration:
    name: "Immigration & Visa"
    default_step: CLARIFICATION   # Agent defaults to asking questions first
    mandatory_variables:
      - nationality
      - visa_type
      - residency_status
    guardrail_keywords:            # Keywords that trigger this topic's guardrail
      fr: ["visa", "titre de séjour", "passeport talent"]
      en: ["visa", "residence permit", "work permit"]
      vi: ["thị thực", "giấy phép cư trú"]
    exemplars:                     # Few-shot examples to guide model behavior
      - input: "Je veux renouveler mon passeport talent chercheur"
        output: |
          **[DONNER]**: Vous devez renouveler 2 mois avant l'expiration.
          **[EXPLIQUER]**: La procédure nécessite une convention d'accueil.
          **[DEMANDER]**: Avez-vous votre convention d'accueil signée par votre employeur ?
    rules:
      - "ALWAYS ask for nationality before giving visa-specific advice."
      - "If user has a 'titre de séjour', ask for its type (10-year/1-year)."
```

---

## Multilingual Keywords (`guardrail_keywords`)

Keywords are organized by language code (`fr`, `en`, `vi`) to make it clear which language each keyword serves and to simplify adding new languages.

```yaml
guardrail_keywords:
  fr: ["grève", "chômage technique", "heures supplémentaires"]
  en: ["strike", "short-time work", "overtime"]
  vi: ["đình công", "lương làm thêm giờ", "hợp đồng lao động"]
```

The `TopicRegistry` Python class flattens all languages into a single keyword index for lookup, maintaining **full backward compatibility** with any topic that still uses the old flat list format.

---

## Few-Shot Exemplar Bank

Exemplars are concrete input/output pairs that guide the model to produce the correct `[DONNER]/[EXPLIQUER]/[DEMANDER]` format. They are more effective than abstract rules because the model pattern-matches on examples rather than interpreting ambiguous instructions.

**Design principle**: If the model produces a wrong answer for a specific case, *add one exemplar* that correctly handles it. No code change required.

---

## How Rules Are Injected

In `orchestrator.py` (Fast Lane) and `procedure_agent.py` (Slow Lane), the registry is used to build the system prompt:

```python
detected_topic = topic_registry.detect_topic(query, intent)
topic_fragment  = topic_registry.build_prompt_fragment(detected_topic, user_profile, query)
global_rules    = topic_registry.build_global_rules_fragment()

system_prompt = f"{topic_registry.persona}\n{topic_fragment}\n{global_rules}"
```

This means the model only ever sees rules relevant to the **current topic**, keeping prompts short and focused.

---

## Adding a New Topic

1. Add a new entry under `topics:` in `topic_registry.yaml`.
2. Add `guardrail_keywords` in `fr`, `en`, `vi` (or just a flat list for quick start).
3. Add `rules` and at least one `exemplar`.
4. No Python changes required — the registry reloads on startup.

---

## Testing

Unit tests for the registry are in `tests/unit/test_registry_multilingual_keywords.py`. They cover:
- Backward compatibility with flat keyword lists.
- Correct topic detection for queries in all 3 languages.
- Keyword index building with multilingual dicts.

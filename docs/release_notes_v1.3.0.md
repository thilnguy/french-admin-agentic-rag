# Release Notes — v1.3.0

**Date:** 2026-02-26  
**Status:** GA (Production Ready) ✅  
**LLM Judge Score:** **9.5/10** 🌟  

---

## 🌟 Executive Summary

v1.3.0 is the final verification release, elevating the system from the 9.1/10 baseline to a **9.5/10 average score** on a 100-case (v3) multilingual blind evaluation. This release focused on stability, multilingual robustness, and eliminating specific failure modes surfaced by the v1 and v2 evals. It also includes a full project cleanup: hardcoded model names removed and all `src/` imports normalized.

---

## 🚀 Key Features

### 1. Multilingual Keyword Dictionary Format

**Problem:** All `guardrail_keywords` in `topic_registry.yaml` were flat lists mixing French, English, and Vietnamese keywords, making it impossible to reason about per-language coverage or add new languages safely.

**Solution:** Converted all 9 topic `guardrail_keywords` to a multilingual dict format:

```yaml
guardrail_keywords:
  fr: ["grève", "heures supplémentaires"]
  en: ["strike", "overtime"]
  vi: ["đình công", "làm thêm giờ"]
```

The `TopicRules` Python class flattens the dict automatically, maintaining **full backward compatibility** with flat lists.

**Impact:** Clear per-language keyword ownership; much easier to audit and extend.

### 2. Vietnamese Query Guardrail Fixes

**Problem:** Vietnamese-language queries about health insurance (`bảo hiểm y tế`) and labor rights (`hợp đồng lao động`) were incorrectly rejected by the topic guardrail because the English keywords didn't match.

**Solution:**
- Added explicit Rule 5 (Labor) and Rule 9 (Healthcare) in `guardrails.py` with Vietnamese exemplar anchor phrases.
- Added Vietnamese keywords to both `labor` and `social` topics in `topic_registry.yaml`.

**Impact:** Vietnamese health/labor queries now correctly pass the guardrail.

### 3. Strike Pay Hallucination Fix

**Problem:** In response to "Can my employer cut my pay during a strike?", the agent incorrectly described French short-time work (`chômage partiel`) rules.

**Root Cause:** The `labor` topic's guardrail keywords were triggering on the wrong policy domain; the prompt lacked a disambiguation exemplar.

**Solution:** Added an exemplar specifically for strike/pay scenarios in `topic_registry.yaml` labor section.

**Impact:** Eliminates the most commonly observed hallucination pattern.

### 4. Identity Keywords Expansion

**Problem:** Queries like "I lost my carte de résident" were not triggering the `identity` topic reliably for non-French speakers.

**Solution:** Expanded the `en` keywords in the identity topic (`residence card`, `foreign national ID`, etc.).

---

## 🛠 Technical Improvements

### Hardcode Elimination (All `src/`)

All hardcoded model names were moved to `src/config.py` settings:

| File | Before | After |
| :--- | :--- | :--- |
| `guardrails.py` | `"gpt-4o-mini"` hardcoded | `settings.GUARDRAIL_MODEL` |
| `preprocessor.py` | `"gpt-4o-mini"` hardcoded | `settings.FAST_LLM_MODEL` |
| `procedure_agent.py` | `"gpt-4o"` in Prometheus label | `self.llm.model_name` (dynamic) |

New settings in `config.py`:
```python
GUARDRAIL_MODEL: str = "gpt-4o-mini"   # configurable via env var
FAST_LLM_MODEL: str = "gpt-4o-mini"    # configurable via env var
```

### Code Cleanup

- Removed dead `self.clarification_prompt = ChatPromptTemplate.from_template("...")` placeholder in `procedure_agent.py`.
- Moved all `from src...import` inline imports in `orchestrator.py` to top-level import block (clean, idiomatic Python).
- Removed stale multi-line code comments from `orchestrator.py` ROUTER section.

---

## 📊 Benchmarks (v3 — 100-case eval, GPT-4o)

| Metric | v1 (9.1/10) | **v3 (9.5/10)** | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Score** | 9.1 | **9.5** | +0.4 |
| **Clarification Accuracy** | ~89% | **~92%** | +3% |
| **Hallucination Rate** | ~2% | **0%** | -2% |
| **Vietnamese Coverage** | ~70% | **~95%** | +25% |
| **Edge Case Score** | ~8.5 | **~9.0** | +0.5 |

**Verdict: ✅ GO for Production**

---

## 📂 New Documentation

- [`docs/rule_system.md`](rule_system.md) — Complete guide to the Topic Registry YAML format, multilingual keywords, and exemplar bank.
- [`docs/release_notes_v1.2.0.md`](release_notes_v1.2.0.md) — Retroactive release notes for v1.2.0.
- [`docs/architecture_evolution.md`](architecture_evolution.md) — Updated to reflect current state (v1.3.0), marking all migration phases as complete.

---

## 🔮 What's Next? (Phase 2)

- **Tracing**: OpenTelemetry integration for distributed query tracing.
- **Monitoring**: Prometheus/Grafana dashboard for latency p95 and token usage.
- **Streaming**: Full SSE streaming for fast token-by-token display in the frontend.

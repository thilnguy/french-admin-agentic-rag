# Release Notes — v1.2.0

**Date:** 2026-02-22  
**Status:** Stable  
**LLM Judge Score:** 9.1/10 (v1 eval) → 8.8/10 (post-guardrail hardening regression, resolved in v1.3.0)

---

## 🌟 Executive Summary

v1.2.0 introduced the **Data-Driven Topic Registry** refactor and the **Few-Shot Exemplar Bank** — transforming a brittle, hardcoded prompt system into a YAML-configurable rule engine. This release also included an experimental **Local Brain** strategy using a fine-tuned Qwen 2.5 7B 8-bit model on Mac M4 (via MLX), and ran the first 100-case blind evaluation dataset.

---

## 🚀 Key Features

### 1. Data-Driven Topic Registry

**Problem:** Admin rules were scattered across 600+ lines of monolithic Python prompt strings. Adding a new topic required editing both the prompt logic and the guardrail code.

**Solution:** Implemented `src/rules/topic_registry.yaml` and `src/rules/registry.py`:
- Each of the 9 topics declares its own `rules`, `mandatory_variables`, `guardrail_keywords`, and `exemplars`.
- The `TopicRegistry.build_prompt_fragment()` injects *only relevant rules* into the prompt (~60% shorter prompts).
- `detect_topic()` uses a keyword index for fast, zero-LLM-cost classification.

**Impact:** Reduced prompt complexity significantly; enabled easy per-topic iteration without touching Python.

### 2. Few-Shot Exemplar Bank

**Problem:** Abstract rules like "use DONNER/EXPLIQUER/DEMANDER format" were inconsistently followed.

**Solution:** Added 2-3 concrete input/output exemplars per topic in `topic_registry.yaml`. The model now pattern-matches on examples rather than interpreting instructions.

**Impact:** Improved format consistency and reduced clarification-avoidance behavior.

### 3. Experimental Local Brain (Qwen 2.5 7B Fine-tuning)

**Goal:** Explore running the main generation model locally on a Mac M4 (MLX) to avoid cloud API costs.

**Process:**
- Curated a 500-item fine-tuning dataset of administrative Q&A pairs.
- Fine-tuned Qwen 2.5 7B in 8-bit quantization using MLX-LM on a Mac M4.
- Evaluated across 20-case and 100-case benchmarks.

**Results:**
- Local model achieved **~7.8/10** on the blind 100-case eval — below the GPT-4o baseline of 9.1/10.
- Key failure modes: format inconsistency, over-refusal on edge cases, and translation quality degradation.

**Decision:** Retained GPT-4o as the production model. Fine-tuning is documented in `finetuning/` as a reference for future iterations.

### 4. 100-Case Blind Evaluation Framework

- Created `evals/data/benchmarks/ds_eval_blind_registry_gpt4o_v1_100.json` with 100 diverse cases across immigration, labor, taxes, identity, social, housing, health, transport, and education topics.
- Cases cover French, English, and Vietnamese queries, including complex scenarios: dual nationality, cross-border work, refugee status.
- Automated grading via an LLM Judge (`evals/runners/llm_judge.py`) using GPT-4o.

---

## 🛠 Technical Improvements

- **Eval Pipeline**: Full JSON result output with per-case scoring, reasoning, and hallucination verdicts.
- **`analyze_eval_results.py`**: Script to compute aggregate statistics from eval JSON files.
- **Guardrail Hardening (initial)**: Expanded guardrail rules to reduce false refusals on Vietnamese and cross-border queries.

---

## 📊 Benchmarks (v1 — 100-case eval, GPT-4o)

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Overall Score** | **9.1/10** | GPT-4o main model |
| **Clarification Accuracy** | ~89% | Appropriate DEMANDER calls |
| **Hallucination Rate** | ~2% | Minor wording issues found |
| **Language Consistency** | 98% | vi/en/fr correctly preserved |

---

## 🐛 Known Regressions

- After guardrail hardening (adding Rule 5/9 for Vietnamese health/labor), re-test showed a drop to **8.8/10** on a subset before the fix was stabilized.
- Root cause: expanded keyword matching introduced some false positives that were later resolved in v1.3.0.

---

## 🔮 What's Next? (v1.3.0)

- Multilingual `guardrail_keywords` dict format (`fr/en/vi` per topic).
- Targeted fixes for "strike pay" hallucination and Vietnamese identity query refusals.
- Final 9.5/10 score validation on 100-case v3 eval.

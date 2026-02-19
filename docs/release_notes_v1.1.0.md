# Release Notes v1.1.0

**Date:** 2026-02-19
**Status:** Stable (Production Ready)
**LLM Judge Score:** 10/10 🌟

## 🌟 Executive Summary
This release marks a major milestone in the French Admin Agentic RAG system, achieving a **perfect 10/10 score** on the LLM Judge benchmark. Key improvements include the introduction of a **Hybrid Search Architecture** (BM25 + Vector), a modular **Query Pipeline**, and specialized **Agentic Workflows** for complex procedures. The codebase has been significantly refactored for maintainability and testability, with legacy code removed and test coverage increased to **91%**.

---

## 🚀 Key Features

### 1. Hybrid Search with RRF Fusion
- **Problem**: Pure vector search struggled with exact keyword matches (e.g., form numbers "Cerfa 12345").
- **Solution**: Implemented a **Hybrid Retriever** combining:
  - **BM25 (Sparse)**: For precise keyword matching.
  - **Qdrant (Dense)**: For semantic understanding.
  - **Reciprocal Rank Fusion (RRF)**: To merge results intelligently.
- **Impact**: Significantly improved retrieval recall for specific administrative queries.

### 2. Intelligent Query Pipeline
- **Refactor**: Replaced the monolithic `handle_query` with a modular `QueryPipeline` and `LanguageResolver`.
- **Components**:
  - `LanguageResolver`: Robust language detection with anti-hallucination logic (prevents "French query answered in Vietnamese").
  - `QueryPipeline`: Standardized `Rewrite -> Intent -> Extract` flow.
  - `Router`: Dynamic routing to "Fast Lane" (Simple QA) or "Slow Lane" (Agent Graph).

### 3. Specialized Agents
- **ProcedureGuideAgent**: Optimized for step-by-step guidance.
  - **Feature**: Automatically asks clarification questions based on priority variables (Nationality > Residence Status).
  - **Feature**: Forces retrieval for fact-checks (e.g., student work hours).
- **LegalResearchAgent**: Enhanced for deep legal queries.

---

## 🛠 Technical Improvements

### 1. Robustness & Safety
- **Anti-Hallucination**: Strict grounding rules in prompts.
- **Event Loop Safety**: Fixed `RuntimeError` in `test_orchestrator.py` by properly handling async memory managers.
- **Behavioral Contract**: Enforced Pydantic validation on `AgentState`.

### 2. Test Suite Overhaul
- **Coverage**: Increased to **91%**.
- **Performance**: Tests run faster due to extensive mocking of heavy components (Redis, OpenAI).
- **Cleanup**: Removed obsolete integration tests from the previous "Layer" architecture.
- **Pass Rate**: **124/124 Tests Passed** (100%).

---

## 📊 Benchmarks (LLM Judge)

| Metric | Score | Status |
| :--- | :--- | :--- |
| **Overall Score** | **10.0/10** | 🏆 Perfect |
| **Response Accuracy** | 100% | ✅ |
| **Retrieval Recall** | High | ✅ (via Hybrid) |
| **Hallucination Rate** | 0% | ✅ |
| **Language Consistency** | 100% | ✅ |

---

## 🔮 What's Next? (v1.2.0)
- **Observability**: Integrate OpenTelemetry for distributed tracing.
- **Deployment**: Kubernetes manifests for scalable production deployment.
- **Voice Interface**: Enhanced real-time voice processing.

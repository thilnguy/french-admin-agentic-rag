# Production Improvements Changelog

> **Project**: french-admin-agentic-rag
> **Status**: Phase 1 Complete ✅ | Phases 2-4 Pending
> **Date**: 2026-02-11

---

## Phase 1: Critical Security & Correctness ✅

**Priority**: 🔴 Blocking
**Status**: Complete
**Files Modified**: 8

### Security Fixes

#### 1. CORS Wildcard Removed
**Files**: [config.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/config.py), [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py)

**Before**:
```python
allow_origins=["*"]  # Any domain can call the API
```

**After**:
```python
# config.py
ALLOWED_ORIGINS: str = "http://localhost:3000"

# main.py
allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
```

**Impact**: Prevents CSRF attacks and unauthorized API access.

---

#### 2. Rate Limiting Added
**Files**: [pyproject.toml](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/pyproject.toml), [config.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/config.py), [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py)

**Added**:
```python
# Dependencies
"slowapi>=0.1.9"

# Config
RATE_LIMIT: str = "10/minute"

# main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit(settings.RATE_LIMIT)
async def chat(request: Request, chat_request: ChatRequest):
    ...
```

**Impact**: Prevents API abuse, protects against DoS, controls OpenAI costs.

---

#### 3. Temp File Vulnerability Fixed
**File**: [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py#L91-L106)

**Before**:
```python
temp_path = f"temp_{audio.filename}"  # Path traversal risk, no cleanup
```

**After**:
```python
import tempfile, os
temp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name
    # ... process audio ...
finally:
    if temp_path and os.path.exists(temp_path):
        os.unlink(temp_path)
```

**Impact**: Prevents path traversal attacks, ensures temp file cleanup.

---

#### 4. TTS Concurrent Safety
**File**: [polyglot_voice/main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/skills/polyglot_voice/main.py#L29-L31)

**Before**:
```python
output_path = "output_voice.mp3"  # Concurrent requests overwrite each other
```

**After**:
```python
import uuid
output_path = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
```

**Impact**: Prevents race conditions in voice responses.

---

### Correctness Fixes

#### 5. Lock File Now Tracked
**File**: [.gitignore](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/.gitignore)

**Before**:
```
uv.lock  # ⚠️ Untracked = non-reproducible builds
```

**After**:
```
# Removed uv.lock from .gitignore
# Added cv_*.tex (personal files)
# Added output_voice.mp3, /tmp/tts_* (temp audio)
```

**Impact**: Ensures reproducible builds across environments.

---

#### 6. Consistent Config Management
**File**: [manager.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/memory/manager.py)

**Before**:
```python
self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
```

**After**:
```python
from src.config import settings
self.redis_url = settings.REDIS_URL
```

**Impact**: All config now centralized in Pydantic settings.

---

#### 7. Fixed Mixed-Language Strings
**Files**: [guardrails.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/shared/guardrails.py#L82-L86), [orchestrator.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/agents/orchestrator.py#L61-L68)

**Before**:
```python
# French disclaimer was mixed FR/VI
"fr": "...Ces thông tin chỉ mang tính chất tham khảo..."

# Hardcoded Vietnamese rejection
return f"Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Lý do: {reason}"
```

**After**:
```python
# Pure French disclaimer
"fr": "...Ces informations sont données à titre indicatif..."

# i18n rejection messages
rejection_messages = {
    "fr": "Désolé, je ne peux pas traiter cette demande. Raison : {reason}",
    "en": "Sorry, I cannot process this request. Reason: {reason}",
    "vi": "Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Lý do: {reason}",
}
lang_key = self.lang_map.get(user_lang.lower(), "French")[:2].lower()
return rejection_messages.get(lang_key, rejection_messages["fr"]).format(reason=reason)
```

**Impact**: Correct language-specific UX for all supported languages.

---

#### 8. FastAPI Lifespan Events
**File**: [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py#L17-L25)

**Added**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — French Admin Agent ready.")
    yield
    logger.info("Shutting down — closing connections...")
    try:
        await orchestrator.cache.aclose()
    except Exception:
        pass
```

**Impact**: Proper async resource cleanup on shutdown.

---

### Dependency Resolution Fix

**File**: [pyproject.toml](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/pyproject.toml)

**Removed**:
```toml
[tool.uv]
index-url = "https://pypi.org/simple"
extra-index-url = ["https://download.pytorch.org/whl/cpu"]
```

**Reason**: These lines caused `uv` to resolve `urllib3`/`requests` from the PyTorch index (old versions), blocking lockfile generation. The `explicit = true` index config alone correctly routes only `torch`/`torchvision` to CPU index.

**Result**: `uv lock` now succeeds, added `slowapi v0.1.9`.

---

## Phase 2: Performance & Code Quality ✅

**Priority**: 🟡 Important
**Status**: Complete
**Files Modified**: 3

### 1. Singleton Qdrant Client + Embeddings
**File**: [legal_retriever/main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/skills/legal_retriever/main.py)

**Before**: New `QdrantClient` + `HuggingFaceEmbeddings` created per request (~seconds each)

**After**:
```python
@lru_cache(maxsize=1)
def _get_qdrant_client():
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
```

**Impact**: First request loads once, all subsequent requests reuse. Added `warmup()` to pre-load during startup.

---

### 2. Singleton Translator LLM
**File**: [admin_translator.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/skills/admin_translator.py)

**Before**: `ChatOpenAI(model="gpt-4o", ...)` per call

**After**: Module-level singleton with lazy init.

---

### 3. Dead Code Removed
**File**: [legal_retriever/main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/skills/legal_retriever/main.py)

Removed lines 20-63 (first `tasks`/`search_results` block that was overridden by the re-implementation below it). File reduced from 102 → 75 lines.

---

### 4. Deep Health Checks
**File**: [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py#L76-L106)

**Before**: Always returned `{"status": "healthy"}`

**After**:
```python
@app.get("/health")
async def health_check():
    redis_ok = redis.from_url(settings.REDIS_URL).ping()
    qdrant_ok = QdrantClient(...).get_collections()
    status = "healthy" if (redis_ok and qdrant_ok) else "degraded"
    return {"status": status, "version": ..., "dependencies": {"redis": ..., "qdrant": ...}}
```

**Impact**: Kubernetes/Docker liveness probes can now detect degraded state.

---

### 5. Startup Warmup
**File**: [main.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/src/main.py#L19-L30)

Lifespan event now calls `warmup_retriever()` on startup, pre-loading the embeddings model and Qdrant client before first user request.

---

## Phase 3: Testing ✅

**Priority**: 🟡 Important
**Status**: Complete
**Result**: 22 tests, all passing in 0.04s

### 1. Guardrail Unit Tests (8 tests)
**File**: [test_guardrails.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/tests/unit/test_guardrails.py) `[NEW]`

| Test | Validates |
|------|-----------|
| `test_validate_topic_admin_question_approved` | Admin question → APPROVED |
| `test_validate_topic_unrelated_rejected` | Cooking question → REJECTED |
| `test_validate_topic_followup_with_history_approved` | "Why?" with admin history → APPROVED |
| `test_check_hallucination_safe` | Grounded answer → SAFE |
| `test_check_hallucination_detected` | Fabricated answer → HALLUCINATION |
| `test_add_disclaimer_french` | FR disclaimer text correct |
| `test_add_disclaimer_english` | EN disclaimer text correct |
| `test_add_disclaimer_unknown_language_defaults_to_french` | Unknown lang → FR fallback |

---

### 2. Retriever Unit Tests (4 tests)
**File**: [test_retriever.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/tests/unit/test_retriever.py) `[NEW]`

| Test | Validates |
|------|-----------|
| `test_retrieve_general_returns_results` | General domain queries both collections |
| `test_retrieve_missing_collection_returns_empty` | Missing collection → `[]` (no crash) |
| `test_retrieve_procedure_domain_only` | Procedure domain → only service-public |
| `test_retrieve_legislation_domain_only` | Legislation domain → only legi |

---

### 3. Translator Unit Tests (3 tests)
**File**: [test_translator.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/tests/unit/test_translator.py) `[NEW]`

| Test | Validates |
|------|-----------|
| `test_translate_returns_translation` | FR→EN translation returns result |
| `test_translate_accepts_vietnamese` | FR→VI translation works |
| `test_translate_passes_correct_params` | Correct `text` + `target_language` passed to chain |

---

### 4. Integration Tests Rewritten (4 tests)
**File**: [test_api.py](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/tests/integration/test_api.py)

**Before**: `test_chat_flow_mocked` accepted `status_code in [200, 500]` — test always passed even when broken.

**After**: All tests properly mock dependencies and assert 200 with correct response structure.

| Test | Validates |
|------|-----------|
| `test_health_check` | `/health` returns status + dependencies (mocked Redis/Qdrant) |
| `test_chat_endpoint_validation` | Missing query → 422, invalid language → 422 |
| `test_chat_flow_mocked` | `/chat` → 200, correct `answer` field (mocked orchestrator) |
| `test_root_endpoint` | `/` returns "French Admin Agent" status |

---

## Phase 4: DevOps & Observability ✅

**Priority**: 🟢 Enhancement
**Status**: Complete
**Files Created**: 4 (`evals/` scripts + `docs/production_roadmap.md`)

### 1. Production Readiness Review
**File**: [production_roadmap.md](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/docs/production_roadmap.md) `[NEW]`

Evaluated system across 7 dimensions (Score: **6.6/10**).
- **Architecture**: 8/10 (Clean async pipeline)
- **Security**: 7/10 (Missing API auth)
- **RAG Quality**: 5/10 (Needs retriever improvement)

Roadmap created for critical security & quality fixes (Phase 1-3).

---

### 2. RAG Evaluation Suite
**Files**: `evals/eval_rag_quality.py`, `evals/eval_performance.py`, `evals/eval_guardrails.py`

Comprehensive evaluation pipeline implemented:

| Component | Metric | Result | Note |
|-----------|--------|--------|------|
| **Faithfulness** | Generator | **76.6%** | Good grounding in context |
| **Hallucination** | Generator | **13.7%** | Needs improvement (<10% target) |
| **Claim Recall** | Retriever | **41.1%** | Major bottleneck (top-k=5 too low) |
| **Precision** | Guardrail | **>95%** | Topic validation is accurate |
| **Latency p95** | Performance | **~200ms** | Retrieval is fast |
| **Latency p95** | E2E | **~3.2s** | LLM generation dominates |

---

### 3. CI/CD Pipeline Review
- **GitHub Actions**: Lint + Test + Docker Build verified
- **Pre-commit hooks**: `ruff` linting enforced
- **Docker**: Multi-stage build with non-root user security best practices

---

## Conclusion
The system is functionally complete and stable (Phase 1-3). The primary focus for production readiness (Phase 4) is now **improving RAG retrieval quality** (recall is low) and **hardening security** (API auth).

See [production_roadmap.md](file:///Users/lananh/Workspace/code/MyAGWorkspace/french-admin-agentic-rag/docs/production_roadmap.md) for next steps.

---

## Phase 1 (Architecture Evolution): State Management & Intent Classification ✅
**Status**: Complete
**Date**: 2026-02-12

### 1. Structured State Management
- **Refactored**: Moved from simple message list to structured `AgentState` (Pydantic model) stored in Redis.
- **File**: `src/agents/state.py`
- **Impact**: Enables complex multi-turn workflows by tracking `user_profile`, `intent`, and `current_step`.

### 2. Intent Classification
- **Implemented**: `IntentClassifier` using `gpt-4o-mini`.
- **Categorization**: `SIMPLE_QA`, `COMPLEX_PROCEDURE`, `FORM_FILLING`.
- **File**: `src/agents/intent_classifier.py`
- **Impact**: Foundation for "Router-First" hybrid architecture.

### 3. Orchestrator Update
- **Integrated**: `AdminOrchestrator` now uses `AgentState` and `IntentClassifier`.
- **File**: `src/agents/orchestrator.py`
- **Tests**: Added `tests/unit/test_state_management.py` and `tests/integration/test_orchestrator_flow.py`.


## Phase 3 (Architecture Evolution): Hybrid Router & AgentGraph ✅
**Status**: Complete
**Date**: 2026-02-12

### 1. AgentGraph Implementation
- **Implemented**: `src/agents/graph.py` defining a LangGraph `StateGraph`.
- **Agents**: Orchestrates `LegalResearchAgent` and `ProcedureGuideAgent`.
- **Routing**: `AgentState.intent` determines the path (`LEGAL_INQUIRY` -> Legal, `COMPLEX_PROCEDURE` -> Procedure).

### 2. Hybrid Router in Orchestrator
- **Updated**: `AdminOrchestrator` delegates complex tasks to `AgentGraph`.
- **Fast Lane**: `SIMPLE_QA` remains on the optimized legacy RAG path.
- **Slow Lane**: `COMPLEX_PROCEDURE`, `FORM_FILLING`, `LEGAL_INQUIRY` use the graph.
- **Verification**: `tests/unit/test_router_integration.py` confirms correct routing logic.
{{ ... }}

### 2. Hybrid Router in Orchestrator
- **Updated**: `AdminOrchestrator` delegates complex tasks to `AgentGraph`.
- **Fast Lane**: `SIMPLE_QA` remains on the optimized legacy RAG path.
- **Slow Lane**: `COMPLEX_PROCEDURE`, `FORM_FILLING`, `LEGAL_INQUIRY` use the graph.
    - [x] Verification: `tests/unit/test_router_integration.py` confirms correct routing logic.

## Phase 5 (Testing & Coverage) ✅
**Status**: Complete
**Date**: 2026-02-12

### 1. Comprehensive Test Suite
- **Goal**: >90% code coverage.
- **Achieved**: **94%** coverage with 55 passing tests.
- **Key Additions**:
    - **Graph Testing**: `tests/unit/test_graph.py` (100% coverage).
    - **Intent Classifier**: `tests/unit/test_intent_classifier.py` (100% coverage).
    - **Orchestrator Exceptions**: Added tests for Redis failures, Guardrail rejections, and Pydantic model mocks.
    - **Main API Exceptions**: `tests/unit/test_main_exceptions.py` covering global handlers.

### 2. Validation Results
All tests passed, including edge cases for error handling and fallback logic.

### Phase 6: Optimization & Security Hardening (Post-Debate) ✅
**Status**: Complete
**Date**: 2026-02-13

### 1. Cost & Latency Optimization
- **Downgraded Models**: Switched `LegalResearchAgent` helper steps (`_refine_query`) to `gpt-4o-mini`.
- **Merged Steps**: Removed explicit `_evaluate_context` call, merging it into `_synthesize_answer`.
- **Impact**: Reduced LLM round-trips from 3 to 2 per query, significantly lowering latency and cost.

### 2. Security Hardening
- **Universal Guardrail**: Applied `check_hallucination` to `AgentGraph` output in `AdminOrchestrator`.
- **Impact**: Ensures "Slow Lane" expert agents are subject to the same strict safety checks as the "Fast Lane".

### 3. Verification
- **Tests**: Updated `tests/unit/test_legal_agent.py` and `tests/unit/test_orchestrator.py` to verify new flows and security stops.

### Phase 7: Real-time Streaming (Post-Debate) ✅
**Status**: Complete
**Date**: 2026-02-13

### 1. Implementation
- **Backend**: Added `stream_query` to `AdminOrchestrator` supporting both Fast Lane (simulated streaming) and Slow Lane (`astar_events`).
- **API**: Added `POST /chat/stream` endpoint returning SSE (`text/event-stream`).

### 2. Verification
- **Test**: Created `tests/integration/test_streaming_endpoint.py` verifying event format and content.
- **Latency**: User perception improved from >3s wait to instant token feedback.

### Phase 8: Final Coverage Push ✅
**Status**: Complete
**Date**: 2026-02-13

### 1. Actions
- **Fix**: Updated `test_router_integration.py` to match new security signatures.
- **New Tests**: Added `tests/unit/test_orchestrator_stream.py` to cover streaming logic (Fast/Slow lane, Caching, Error handling).

### 2. Final Coverage Report
| Module | Coverage | Status |
| :--- | :--- | :--- |
| `src.agents.orchestrator` | **91%** | ✅ |
| `src.agents.legal_agent` | **95%** | ✅ |
| `src.main` | **89%** | ✅ |
| **TOTAL PROJECT** | **93%** | **PASSED** (>90%) |

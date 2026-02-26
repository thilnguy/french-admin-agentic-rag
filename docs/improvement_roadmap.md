# Kế Hoạch Cải Thiện theo Mức Độ Ưu Tiên

**Ngày:** 2026-02-26  
**Phiên bản hiện tại:** v1.3.0 (Production Maturity: ~6.5/10)

---

## Ma Trận Ưu Tiên

Mỗi hạng mục được đánh giá theo:
- **Impact**: Tác động đến chất lượng/reliability thực tế
- **Effort**: Công sức ước tính
- **Priority**: Tổng hợp (P1 = làm ngay, P2 = tháng này, P3 = Q2-Q3 2026)

---

## 🔴 P1 — Làm Ngay (tuần 1-2)
*Những thứ mà nếu thiếu sẽ gây lỗi không debug được hoặc dữ liệu lỗi thời*

### 1.1 Distributed Tracing (OpenTelemetry)
- **Vấn đề**: Khi agent graph fail, không có cách trace xem lỗi ở node nào (ProcedureGuide? LegalAgent? Guardrail?). Hiện tại là "fly blind".
- **Giải pháp**: Tích hợp OpenTelemetry + Jaeger/OTLP với custom spans cho mỗi node trong LangGraph.
- **Impact**: Cao — giảm debugging time từ giờ xuống phút
- **Effort**: 2-3 ngày
- **Files**: `src/utils/`, `src/agents/orchestrator.py`, `src/agents/graph.py`

### 1.2 Legal Data Update Pipeline
- **Vấn đề**: Knowledge base (Qdrant) là static. Khi pháp luật thay đổi (e.g., visa quota mới, thuế mới), agent trả lời theo thông tin cũ mà người dùng không biết.
- **Giải pháp**: Script tự động crawl/update từ `service-public.fr` + `legifrance.gouv.fr` với versioning, chạy weekly qua CI.
- **Impact**: Rất cao — đây là critical flaw của bất kỳ legal RAG nào
- **Effort**: 3-5 ngày
- **Files**: `scripts/`, `.github/workflows/`

### 1.3 Graceful Degradation khi Infrastructure Down
- **Vấn đề**: Nếu Qdrant down, agent crash toàn bộ thay vì trả lời với fallback message.
- **Giải pháp**: Try/except quanh `retrieve_legal_info()` với fallback response "Tôi không thể truy cập dữ liệu ngay lúc này, vui lòng thử lại sau."
- **Impact**: Cao — UX và reliability
- **Effort**: 4 giờ
- **Files**: `src/agents/orchestrator.py`, `skills/legal_retriever/main.py`

---

## 🟠 P2 — Tháng Này (tuần 3-4)
*Những thứ cần có để system đáng tin cậy trong production*

### 2.1 Prometheus/Grafana Dashboard
- **Vấn đề**: Metrics được emit (`LLM_REQUEST_DURATION`, `LLM_TOKEN_USAGE`) nhưng không có dashboard để visualize. `prometheus.yml` đã có nhưng chưa kết nối Grafana.
- **Giải pháp**: Tạo Grafana dashboard template với: latency p50/p95/p99, token usage per model, guardrail rejection rate, cache hit rate.
- **Impact**: Trung bình-cao — biết khi nào system chậm hoặc tốn kém
- **Effort**: 1-2 ngày

### 2.2 Streaming Response (SSE)
- **Vấn đề**: Hiện tại user phải chờ toàn bộ response (5-30s cho complex queries) mới thấy gì, UX rất kém.
- **Giải pháp**: FastAPI `StreamingResponse` + `ChatOpenAI(streaming=True)` + frontend EventSource.
- **Impact**: Cao về UX — perceived latency giảm 80%
- **Effort**: 2-3 ngày
- **Files**: `src/main.py`, `src/agents/orchestrator.py`, frontend

### 2.3 Audit Logging cho Queries Nhạy Cảm
- **Vấn đề**: Không có log cụ thể nào ghi lại những query nhạy cảm (visa rejection, tax advice) để retrospective review.
- **Giải pháp**: Structured log với session_id, topic, intent, score (nếu có) — export sang Elasticsearch hoặc tệp log riêng.
- **Impact**: Trung bình — compliance và quality improvement
- **Effort**: 1 ngày

### 2.4 Human Expert Evaluation (Ground Truth Dataset)
- **Vấn đề**: LLM Judge bias (GPT-4o judge GPT-4o). Cần ground truth từ người thực sự hiểu luật hành chính Pháp.
- **Giải pháp**: Tạo 50-case ground truth dataset với expert annotation (có thể là người Pháp thực tế hoặc luật sư). Chạy lại eval so sánh.
- **Impact**: Rất cao về độ tin cậy của benchmark
- **Effort**: 1 tuần (phần lớn là coordination, không phải coding)

---

## 🟡 P3 — Q2 2026 (1-3 tháng)
*Scaling và ecosystem*

### 3.1 BGE-Reranker tích hợp vào Hybrid Retrieval
- **Vấn đề**: BM25 + Vector + RRF đã tốt (~85% recall), nhưng precision còn thấp (trả về nhiều documents không liên quan).
- **Giải pháp**: Thêm cross-encoder reranker (BGE-Reranker-v2) trước khi feed context vào LLM.
- **Expected Impact**: Retrieval precision 85% → 92%+
- **Effort**: 2-3 ngày

### 3.2 Prompt Injection Detection Layer
- **Vấn đề**: Hiện tại không có bảo vệ chống adversarial inputs như "Ignore all previous instructions and...".
- **Giải pháp**: Dedicated classifier (nhẹ, có thể regex + small model) trước bước guardrail.
- **Impact**: Trung bình — security hardening
- **Effort**: 2-3 ngày

### 3.3 Kubernetes Production Deployment
- **Vấn đề**: Docker single-instance. Không scale, không rolling update, không auto-heal.
- **Giải pháp**: Kubernetes manifests với: Deployment + HPA (autoscaling), PVC cho Qdrant data, ConfigMap cho settings, Ingress + TLS.
- **Impact**: Cao cho production deployment thực sự
- **Effort**: 1 tuần

### 3.4 Conversation Quality Feedback Loop
- **Vấn đề**: Không có mechanism để collect user feedback (thumbs up/down) và đưa vào cải thiện.
- **Giải pháp**: Feedback endpoint + storage + weekly analysis script → update exemplars trong YAML nếu pattern thất bại được phát hiện.
- **Impact**: Cao về long-term quality improvement
- **Effort**: 3-4 ngày

---

## 🌠 P4 — Tương lai (Q3-Q4 2026)
*Nice-to-have và vision dài hạn*

| Feature | Mô tả | Notes |
| :--- | :--- | :--- |
| **Voice Interface** | WebSocket real-time voice-to-voice | Whisper STT + TTS |
| **Document Upload** | User upload visa/contract PDF → phân tích | Complex: parsing + privacy |
| **Appointment Booking** | Tích hợp calendly/public service API | Requires gov API access |
| **FinOps Dashboard** | Track cost per query, per topic, per model | Important nếu scale lớn |
| **Multi-country Expansion** | Thêm luật Bỉ, Thụy Sĩ (francophone) | Data collection challenge |

---

## Tóm Tắt Roadmap

```
Tuần 1-2  [P1]: Tracing + Data Pipeline + Graceful Degradation
Tuần 3-4  [P2]: Dashboard + Streaming + Audit Log + Human Eval
Tháng 2-3 [P3]: Reranker + Injection Guard + Kubernetes
Q3+ 2026  [P4]: Voice, Document Upload, Multi-country
```

**Sau P2:** Production Maturity → **8.5/10**  
**Sau P3:** Production Maturity → **9.0/10** (Public-facing ready)

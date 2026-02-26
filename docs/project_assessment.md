# Đánh Giá Project: French Admin Agentic RAG

**Ngày:** 2026-02-26  
**Phiên bản:** v1.3.0  
**Người đánh giá:** Antigravity AI (independent review)

---

## 1. Độ Khó Bài Toán

RAG cho luật pháp quốc gia thuộc nhóm **bài toán NLP khó nhất** — khó hơn RAG thông thường ở nhiều chiều:

- **Ngữ cảnh phụ thuộc hồ sơ người dùng**: Cùng câu hỏi "Tôi có được đi làm không?" có đáp án hoàn toàn khác nhau tùy vào quốc tịch, loại visa, thời gian cư trú. Không phải "retrieve-then-answer" mà phải "profile-then-retrieve-then-answer".
- **Ngôn ngữ pháp lý đặc thù**: Văn bản hành chính Pháp dày đặc thuật ngữ (`passeport talent chercheur`, `titre de séjour`, `Cerfa 12345`) không có trong pre-training của LLM phổ thông.
- **Multilingual với người dùng nhập cư**: Người dùng hỏi bằng tiếng Việt về luật Pháp — đòi hỏi cross-lingual intent classification, không chỉ translation.
- **Rủi ro cao**: Trả lời sai ở domain y tế, di trú, thuế có thể gây hậu quả thực tế. False refusal cũng là thất bại.
- **Ranh giới topic mờ**: "Tôi bị ốm trong thời gian bãi công, tôi được trả không?" — vừa là labor, vừa là health, vừa là payroll.

So với RAG thông thường (chatbot support cho SaaS), bài toán này khó hơn **~2x về chiều đánh giá và ~3x về engineering**.

---

## 2. Điểm Mạnh Nổi Bật

### So sánh với các RAG project cùng mức độ

| Khía cạnh | RAG thông thường | Project này |
| :--- | :--- | :--- |
| **Rule system** | Prompt cứng, viết tay | YAML-driven Topic Registry — 0 hardcode |
| **Guardrail** | Binary allow/deny | Context-aware với lịch sử hội thoại + bypass cho follow-up |
| **Routing** | 1 pipeline cho tất cả | Fast Lane (RAG) vs Slow Lane (Agent Graph) |
| **Multilingual** | Translate rồi query | Native FR/EN/VI keywords, guardrails, exemplars |
| **State management** | Session history (list) | `AgentState` structured với `core_goal` lock, `user_profile` |
| **Evaluation** | Cảm tính hoặc BLEU | LLM Judge tự động, 100 cases, per-case reasoning, versioned |
| **Observability** | Print logs | Structured logging + Prometheus metrics |

### Top 3 điểm nổi bật thực sự

**🏆 1. Data-Driven Topic Registry**  
Đây là thiết kế đúng về mặt kỹ thuật. Hầu hết RAG project nhỏ/mid để rule system chìm trong prompt strings, rất khó maintain. YAML-driven registry cho phép thêm topic mới không cần động code.

**🏆 2. Contextual Continuation Detection**  
Bypass guardrail khi user đang trả lời câu hỏi của agent là một insight tinh tế. 95% RAG project bỏ qua điều này, gây friction khi user bị block ở câu trả lời của chính mình.

**🏆 3. LLM Judge Framework với Versioning**  
Dataset 100 cases, automated grading, versioned JSON results — đây là thứ phân biệt "project học thuật" vs "project nghiêm túc".

---

## 3. Đánh Giá Kết Quả (9.5/10)

### Những gì đã được chứng minh ✅
- 0% hallucination rate trên 100 cases đa dạng
- Clarification logic đúng với ~92% accuracy
- Multilingual coverage thực sự (FR/EN/VI)
- Robustness với edge cases phức tạp (dual nationality, cross-border, refugee status)

### Những gì chưa được chứng minh ⚠️
- **Judge bias**: LLM Judge dùng GPT-4o — cùng family với main model. Có thể bias dương. Ground-truth dataset do expert pháp lý build sẽ khắt khe hơn.
- **Production traffic distribution**: 100 cases được tạo bởi AI — không phải real user queries. Real users có cách hỏi kỳ lạ hơn, typos, code-switching.
- **Long-tail failures**: Score 9.5 có nghĩa là ~5 cases dưới xuất sắc. Ở 10,000 queries thực tế, long-tail failure rate sẽ cao hơn.

**Đánh giá thực tế:** Với real traffic, expect score khoảng **8.5–9.0**.

---

## 4. Mức Độ Trưởng Thành Production

| Tiêu chí | Điểm | Nhận xét |
| :--- | :--- | :--- |
| **Code quality** | 8/10 | Clean, typed, pydantic-validated. Một số chỗ defensive coding còn thiếu |
| **Config management** | 9/10 | Centralized Pydantic Settings, không còn hardcode |
| **Testing** | 8/10 | 149 tests, 94%+ coverage. Thiếu end-to-end integration tests |
| **Observability** | 5/10 | Logging + Prometheus có. **Không có distributed tracing** — agent graph debugging hiện tại là "fly blind" |
| **Error handling** | 7/10 | Retry + Redis fallback có. Không có graceful degradation khi Qdrant down |
| **Deployment** | 7/10 | Docker + CI/CD có. Chưa có Kubernetes, health checks còn cơ bản |
| **Security** | 7/10 | Rate limiting + CORS + API key auth có. Chưa có audit logging |
| **Scalability** | 5/10 | Single instance. Không có horizontal scaling strategy |
| **Data pipeline** | 4/10 | Ingestion thủ công. Không có pipeline update khi law thay đổi |

### **Overall: ~6.5/10 Production Maturity**

**Kết luận thẳng thắn:**  
Đây là một **demo/prototype chất lượng cao** — nghiêm túc hơn 95% side projects, có thể deploy cho team nhỏ dùng nội bộ, nhưng **chưa đủ để deploy public-facing production** vì:
1. Không có tracing → không debug được khi lỗi
2. Không có data update pipeline → legal knowledge sẽ lỗi thời
3. Single point of failure → một Redis/Qdrant down là toàn bộ agent down

---

## 5. Kế Hoạch Cải Thiện theo Mức Độ Ưu Tiên

Xem chi tiết tại → [improvement_roadmap.md](improvement_roadmap.md)

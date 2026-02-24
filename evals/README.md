# RAG Evaluation Suite

Comprehensive evaluation framework measuring quality, performance, and guardrail effectiveness.

## 🎯 Evaluation Dimensions

| Dimension | Script | Metrics | Target |
|-----------|--------|---------|--------|
| **Quality** | `eval_rag_quality.py` | Faithfulness, Relevancy, Precision, Recall, Correctness | >0.8-0.9 |
| **Performance** | `eval_performance.py` | E2E latency (p95), Retrieval latency | <2s, <500ms |
| **Guardrails** | `eval_guardrails.py` | Precision, Recall, FPR, FNR | >0.95, >0.98 |

## 📁 Test Data

- `test_data/ds_golden_v1_raw.json` (10 cases): Initial French admin Q&A set.
- `test_data/ds_golden_v2_enriched.json` (30 cases): Expanded set with expected behavior metadata.
- `test_data/ds_eval_9.8_blind_v1.json` (20 cases): Core logic benchmark (Achieved 9.8/10).
- `test_data/ds_adversarial_guardrails.json` (10 cases): Edge cases for factual traps and off-target queries.
- `test_data/ds_bench_latency.json` (5 cases): Latency benchmarks.

## 🚀 Running Evaluations

**Prerequisites:**
```bash
docker-compose up -d redis qdrant
export OPENAI_API_KEY=sk-...
```

**Run evaluations:**
```bash
cd evals
uv run python eval_rag_quality.py      # Ragas metrics
uv run python eval_performance.py      # Latency benchmarks
uv run python eval_guardrails.py       # Topic validation accuracy
```

**Run all:**
```bash
uv run python eval_rag_quality.py && uv run python eval_performance.py && uv run python eval_guardrails.py
```

## 📊 Metrics

**RAG Quality (Ragas):**
- Faithfulness (>0.9): No hallucinations
- Answer Relevancy (>0.85): Answers match queries
- Context Precision (>0.8): Retrieved docs are relevant
- Context Recall (>0.85): All necessary info retrieved
- Answer Correctness (>0.8): Semantic + factual accuracy

**Performance:**
- E2E p95 <2s, Retrieval p95 <500ms

**Guardrails:**
- Precision >0.95 (low false approvals)
- Recall >0.98 (low false rejections)

## 🔧 Customization

Add test cases to `test_data/golden_set.json`, adjust thresholds in eval scripts.

## 📚 References

- [Ragas Docs](https://docs.ragas.io/)
- [LangSmith](https://docs.smith.langchain.com/) (set `LANGCHAIN_TRACING_V2=true`)

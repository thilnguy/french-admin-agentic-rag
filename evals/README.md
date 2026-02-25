# RAG Evaluation Suite

Comprehensive evaluation framework measuring quality, performance, and guardrail effectiveness.

## 📁 Directory Structure

Organized for scalability and clarity:

- **`data/`**: Datasets categorized by stage
  - `raw/`: Initial golden sets (v1, v2).
  - `enriched/`: Datasets with reasoning and behavior metadata.
  - `benchmarks/`: Adversarial, latency, and blind test sets.
- **`runners/`**: Core evaluation scripts.
- **`utils/`**: Helper tools (enrichment, data cleaning).
- **`results/`**: Execution outputs and judge verdicts.

## 🎯 Evaluation Dimensions

| Dimension | Script | Metrics | Target |
|-----------|--------|---------|--------|
| **Quality** | `runners/eval_rag_quality.py` | Faithfulness, Recall, Precision | >0.8-0.9 |
| **Performance** | `runners/eval_performance.py` | E2E latency (p95), Retrieval latency | <2s, <500ms |
| **Guardrails** | `runners/eval_guardrails.py` | Precision, Recall, FPR, FNR | >0.95, >0.98 |
| **Expert Judge**| `runners/llm_judge.py` | Reasoning score, Hallucination | 10/10 |

## 🚀 Running Evaluations

**Prerequisites:**
```bash
docker-compose up -d redis qdrant
export OPENAI_API_KEY=sk-...
```

**Run evaluations:**
```bash
# Run judge on blind test (default)
uv run python evals/runners/llm_judge.py --limit 5

# Run performance benchmarks
uv run python evals/runners/eval_performance.py

# Run guardrail validation
uv run python evals/runners/eval_guardrails.py
```

## 📊 Metrics

**RAG Quality:**
- Faithfulness (>0.9): No hallucinations
- Relevance (>0.85): Answers match queries
- Precision (>0.8): Retrieved docs are relevant

**Performance:**
- E2E p95 <2s, Retrieval p95 <500ms

**Guardrails:**
- Precision >0.95 (low false approvals)
- Recall >0.98 (low false rejections)

## 🔧 Maintenance

- To enrich new raw data: `uv run python evals/utils/enrich_golden_set.py`
- Results are stored in `evals/results/llm_judge_results_qwen2.5_7B_8bit.json`.

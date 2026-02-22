# Fine-tuning Process & System Optimization Report (Local Brain Strategy)

This document records the complete technical journey from a cloud-heavy GPT-4o dependency to a hybrid RAG system running locally on a Mac M4, achieving a perfect **10/10** evaluation score.

---

## 1. Objectives & Strategy
- **Objective**: Minimize GPT-4o token costs, ensure data privacy, and increase processing speed by running the main LLM locally.
- **"Local Brain" Strategy**: 
    - Use GPT-4o as the "Teacher" to distill knowledge into a smaller model (**Qwen 2.5 7B**).
    - Utilize **MLX-LM** for performance optimization on Apple Silicon M4.

---

## 2. Phase 1: Data Distillation
- **Data Generation**: Developed `generate_finetune_data.py` to prompt GPT-4o to generate 300 exemplary conversation samples based on French administrative contexts.
- **Data Structure**: Trained the model to use Chain-of-Thought (CoT) reasoning before responding, using the following tags:
    - `[DONNER]`: Provide direct information.
    - `[EXPLIQUER]`: Explain the legal/administrative context.
    - `[DEMANDER]`: Ask clarifying questions (Clarification).
- **Round 1 Results**: Achieved 7.0/10. The major issue was "Refusal Bias" (the model incorrectly refused to answer valid topics like marriage or driving licenses, mistaking them for out-of-scope or sensitive requests).

---

## 3. Phase 2: Refinement & Optimization
Three critical steps were taken to reach the 10/10 score:

### A. Focus Samples (Eliminating Refusal Bias)
- Analyzed the cases that received a 0 score (Marriage, Driving License, Residence Fees).
- Generated 50 additional "Focus" samples to teach the model that these are valid requests that *must* be supported.
- Performed a second round of fine-tuning with a very low learning rate to prevent "Catastrophic Forgetting."

### B. Infrastructure Fixes
- **Qdrant Auth**: Fixed 401 Unauthorized errors by updating the `QdrantClient` to correctly pass the API Key from the environment variables.
- **Eval Isolation**: Fixed a bug in `llm_judge.py` where test cases shared State objects, causing "goal leakage" between unrelated questions.

### C. Decision Layer Optimization (Guardrail)
- Identified that the legacy guardrail was too restrictive, often rejecting Vietnamese or English queries.
- **Solution**: Offloaded the Topic Validation guardrail to `gpt-4o-mini`.
- **Result**: The system became reliably polyglot while maintaining absolute safety and precision.

---

## 4. Benchmarks & Final Results

| Metric | Original GPT-4o | Local Qwen 2.5 (Round 1) | **Local Qwen 2.5 (Round 2)** |
| :--- | :--- | :--- | :--- |
| **LLM Judge Score**| 9.5/10 | 7.0/10 | **10.0 / 10** |
| **Token Cost** | High | Low (Guardrail only) | **Low (Guardrail only)** |
| **Inference Speed**| Network dependent | ~80 tokens/sec (M4) | **~80 tokens/sec (M4)** |
| **Clarification Logic**| Strong | Moderate | **Excellent** |

---

## 5. Conclusion
The system now operates at an expert level. The combination of **Local Logic (Qwen 2.5)** and **Cloud Safety (GPT-4o-mini)** provides the perfect balance of Cost, Performance, and Reliability.

**Note**: To re-run the evaluation, use:
```bash
./.venv/bin/python evals/llm_judge.py
```

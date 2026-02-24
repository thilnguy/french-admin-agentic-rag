# Fine-tuning Workflow: French Administration Assistant

This document provides a technical overview of the fine-tuning pipeline, data lineage, and execution scripts used to optimize the model for French administrative procedures.

## 1. Pipeline Overview

The fine-tuning process follows a "Distillation & Augmentation" approach:
1.  **Seed Data Extraction**: High-quality examples are taken from the `golden_set`.
2.  **Self-Instruct Expansion**: LLM (GPT-4o) generates diverse scenarios based on seeds.
3.  **Refusal Correction (Focus Data)**: Targeted generation for topics where the base model previously failed.
4.  **Formatting**: Raw data is converted into ChatML format with `<thinking>` tags.
5.  **Training**: Model is fine-tuned using QLoRA (Axolotl for Cloud/GPU, MLX-LM for Mac M4).

## 2. Data Lineage

All data resides in `finetuning/data/`.

| File | Type | Description |
| :--- | :--- | :--- |
| `train_expert_formatted.jsonl` | **Final Training** | The formatted file used by training scripts. |
| `final_train_raw.jsonl` | Raw | Combined output of self-instruct and focus generation. |
| `train_augmented_500.jsonl` | Augmented | 500 high-quality variations generated from enriched golden set. |
| `self_instruct_samples.jsonl` | Component | ~300 samples covering Basic, Clarify, and Complex scenarios. |
| `focus_samples.jsonl` | Component | ~50 samples targeting specific "hard" topics (e.g., costs, taxes). |
| `valid.jsonl` | Validation | Small slice of data for evaluation during training. |

## 3. Scripts (`finetuning/scripts/`)

| Script | Purpose | Input | Output |
| :--- | :--- | :--- | :--- |
| `self_instruct_expansion.py` | Expands seed data into diverse scenarios. | `evals/data/enriched/ds_golden_v1_enriched.json` | `self_instruct_samples.jsonl` |
| `generate_focus_data.py` | Targets specific topics causing refusals. | `FOCUS_TOPICS` list in script | `focus_samples.jsonl` |
| `create_augmented_finetune_data.py` | Generates 500 high-quality variations. | `evals/data/enriched/ds_golden_v2_enriched.json` | `train_augmented_500.jsonl` |
| `format_for_training.py` | Converts raw JSONL to OpenAI ChatML with CoT. | `final_train_raw.jsonl` | `train_expert_formatted.jsonl` |

## 4. Training Configurations

### Cloud / GPU (NVIDIA)
- **Tool**: [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- **Config**: `finetuning/axolotl_config.yml` (QLoRA, 4-bit, Rank 32)
- **Execution**: `bash finetuning/run_training.sh`

### Local (Mac M4 / Apple Silicon)
- **Tool**: [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llm/mlx_lm)
- **Execution**: `bash finetuning/run_training_m4.sh`
- **Adapters**: Saved to `finetuning/adapters.safetensors`

## 5. Maintenance & Updates

To update the model with new examples:
1.  Add new examples to the relevant seed file or topic list.
2.  Re-run the generation scripts.
3.  Combine raw outputs into `final_train_raw.jsonl`.
4.  Run `python finetuning/scripts/format_for_training.py`.
5.  Re-run the training script of choice.
6.  Verify performance using `uv run python evals/llm_judge.py`.

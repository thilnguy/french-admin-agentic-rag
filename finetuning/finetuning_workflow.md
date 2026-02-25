# Fine-tuning Workflow: French Administration Assistant

This document provides a technical overview of the fine-tuning pipeline, data lineage, and execution scripts used to optimize the model for French administrative procedures.

## 1. Pipeline Overview

The fine-tuning process follows a "Distillation & Augmentation" approach:
1.  **Seed Data Extraction**: High-quality examples are taken from the `golden_set`.
2.  **Self-Instruct Expansion**: LLM (GPT-4o) generates diverse scenarios based on seeds.
3.  **Refusal Correction (Focus Data)**: Targeted generation for topics where the base model previously failed.
4.  **Clarification Injection**: Generating cases where the assistant must ask clarifying questions.
5.  **Formatting**: Raw data is converted into ChatML format with `<thinking>` tags.
6.  **Training**: Model is fine-tuned using QLoRA (MLX-LM for Mac M4).

## 2. Data Lineage

All data resides in `finetuning/data/`.

| File | Type | Description |
| :--- | :--- | :--- |
| `train_raw_v1_350.jsonl` | Raw | V1 base training dataset containing 350 samples. |
| `train_raw_v2_600.jsonl` | Raw | V2 augmented training dataset with 600 total samples, including clarification injection. |

## 3. Scripts (`finetuning/scripts/`)

| Script | Purpose |
| :--- | :--- |
| `generate_training_data.py` | A unified script capable of running multiple generation strategies (`augment`, `clarify`, `focus`, `self-instruct`) to generate synthetic data. parameterized with `argparse`. |
| `format_for_training.py` | Converts raw JSONL to OpenAI ChatML with CoT tags. Parameterized to accept `--input` and `--output`. |

## 4. Training Configurations

### Local (Mac M4 / Apple Silicon)
- **Tool**: [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llm/mlx_lm)
- **Execution**: `bash finetuning/run_training_m4.sh`
- **Adapters**: Saved to `finetuning/adapters.safetensors`

## 5. Maintenance & Updates

To update the model with new examples:
1.  Add new examples to the relevant seed file or topic list.
2.  Run `generate_training_data.py` with the appropriate `--strategy` to generate your samples.
3.  Add the generated samples to your raw working file (e.g. `train_raw_v2_600.jsonl` or an updated v3 file).
4.  Run `python finetuning/scripts/format_for_training.py` to format the new raw dataset into ChatML.
5.  Re-run `bash finetuning/run_training_m4.sh` passing your formatted output.
6.  Verify performance using `uv run python evals/runners/llm_judge.py`.

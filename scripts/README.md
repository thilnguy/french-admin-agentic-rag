# Scripts for French Admin Agentic RAG

This directory contains utility scripts for data preparation, ingestion, and testing.

## Data Preparation (Fine-tuning)

- `self_instruct_expansion.py`: Implements a 3-phase Self-Instruct pipeline (Diversity -> Consistency -> Complexity) to expand seed administrative scenarios into a high-quality expert dataset.
- `format_for_training.py`: Converts raw expert samples into OpenAI ChatML format with reasoning (CoT) traces for Axolotl fine-tuning.

## Infrastructure & Testing

- `ingest_data.py`: Handles vector storage ingestion for Legal RAG.
- `test_agent.py`: CLI tool for interactive testing of the Orchestrator.
- `test_memory.py`: Validates Redis-based session memory longevity.

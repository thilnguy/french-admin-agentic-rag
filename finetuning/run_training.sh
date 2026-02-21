#!/bin/bash
# Helper script to launch fine-tuning for French Admin Assistant

set -e

echo "🚀 Starting Fine-tuning Preparation..."

# 1. Check for GPU
if ! command -v nvidia-smi &> /dev/null
then
    echo "⚠️  WARNING: No NVIDIA GPU detected. Training will be extremely slow (or fail)."
fi

# 2. Check for dataset
DATA_FILE="finetuning/data/train_expert_formatted.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Dataset not found at $DATA_FILE"
    exit 1
fi

# 3. Prompt for Axolotl installation if missing
if ! command -v axolotl &> /dev/null
then
    echo "📦 Axolotl not found. Installing..."
    pip install torch torchvision torchaudio
    pip install "axolotl[flash-attn,preprocess] @ git+https://github.com/OpenAccess-AI-Collective/axolotl"
fi

# 4. Launch Training
echo "🔥 Launching SFT (Target: Qwen 2.5 7B)..."
accelerate launch -m axolotl.cli.train finetuning/axolotl_config.yml

echo "✅ Training complete! Model saved in finetuning/qwen-7b-french-admin-distilled"

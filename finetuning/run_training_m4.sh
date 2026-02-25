#!/bin/bash
# Helper script to launch fine-tuning for French Admin Assistant on Mac M4 (Apple Silicon)

set -e

echo "🚀 Starting Mac M4 (Apple Silicon) Fine-tuning Preparation..."

# 1. Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  WARNING: This script is optimized for Apple Silicon (M1/M2/M3/M4). Proceeding anyway..."
fi

# Accept arguments with defaults
SOURCE_DATA=${1:-"finetuning/data/train_expert_formatted.jsonl"}
ADAPTER_PATH=${2:-"finetuning/adapters.safetensors"}
MODEL=${3:-"mlx-community/Qwen2.5-7B-Instruct-8bit"}
ITERS=${4:-600}

# MLX-LM expects train.jsonl / valid.jsonl in the data directory
DATA_DIR=$(dirname "$SOURCE_DATA")
TARGET_TRAIN="$DATA_DIR/train.jsonl"
TARGET_VALID="$DATA_DIR/valid.jsonl"

if [ ! -f "$SOURCE_DATA" ]; then
    echo "❌ Error: Expert dataset not found at $SOURCE_DATA"
    exit 1
fi

echo "📂 Preparing data for MLX-LM..."
cp "$SOURCE_DATA" "$TARGET_TRAIN"
# Use a small slice for validation
head -n 15 "$SOURCE_DATA" > "$TARGET_VALID"

# 3. Install MLX-LM if missing
echo "📦 Ensuring MLX-LM is available via uv..."
# We use uv to run the training to ensure environment consistency
# This avoids the "ModuleNotFoundError" by using the same interpreter for install and run

# 4. Launch Training
echo "🔥 Launching LoRA Fine-tuning (MLX-LM + Qwen 2.5 7B)..."

uv run --with mlx-lm python -m mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --iters "$ITERS" \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --steps-per-report 10 \
    --steps-per-eval 50 \
    --grad-checkpoint \
    --max-seq-length 1024 \
    --adapter-path "$ADAPTER_PATH"


echo "✅ Training complete! Adapters saved in $ADAPTER_PATH"
echo "💡 To use this model, you can merge or load the adapters in inference."

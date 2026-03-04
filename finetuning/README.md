# Fine-tuning Guide for French Admin Assistant

This is the guide for executing Phase 3 and Phase 4 of the LLM cost optimization plan.

## 1. Data Preparation
The expert dataset has been prepared with 600 samples (including CoT Reasoning and Clarification Injection data):
`finetuning/data/train_expert_formatted.jsonl`


## 2. Hardware Recommendations
### For Linux (NVIDIA GPU)
- **GPU**: Minimum 24GB VRAM recommended (RTX 3090, 4090, A10g, A100) to run QLoRA with a good batch size.
- **RAM**: 32GB+.

### For Mac (Apple Silicon)
- **Chip**: M2/M3/M4 Max or Ultra (high memory bandwidth).
- **Unified Memory**: Minimum 32GB (64GB+ recommended).

---

## 3. Training Environment Setup
We recommend using **Axolotl** or **Unsloth** for maximum speed on Linux, and **MLX** on Mac.

### Linux (NVIDIA)
It is recommended to use a Docker image with pre-installed CUDA toolkit from [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) or use the [Unsloth](https://github.com/unslothai/unsloth) environment to double training speed and reduce VRAM usage by 50%:
```bash
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 4. Running the Training
Use the prepared configuration file:
```bash
axolotl finetuning/axolotl_config.yml
```
Upon completion, the model (LoRA adapters) will be saved in the `finetuning/qwen-7b-french-admin-distilled` directory.

---

## 5. Model Deployment and Usage

Marianne AI supports the two largest local ecosystems currently available: vLLM (for Linux/NVIDIA) and MLX (for Mac/Apple Silicon).

### Option A: Linux Server (Using vLLM)
vLLM is currently the fastest inference engine for NVIDIA production servers.

**Step 1: Run vLLM Server**
```bash
# Install vLLM
pip install vllm

# Launch OpenAI-compatible server with the fine-tuned model (injecting LoRA adapters)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules french_admin=finetuning/qwen-7b-french-admin-distilled \
    --port 8020
```

### Option B: MacBook (Using MLX Server)
For Macs, Apple's MLX framework is the 100% optimal choice.

**Step 1: Run MLX Server**
```bash
uv run --with mlx-lm python -m mlx_lm.server \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --adapter-path finetuning/adapters.safetensors \
    --port 8020
```

---

### Step 2: Agent Configuration (Common for Linux & Mac)
Update the `.env` file in the root directory to point the app requests to the Local Server instead of the Cloud API:
```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:8020/v1
LOCAL_LLM_MODEL=your_custom_model_name_for_mlx_or_vllm
```

### Step 3: Quality Evaluation
To see how the new model performs compared to GPT-4o in your local environment, run the evaluation script:
```bash
uv run python evals/llm_judge.py
```

---

## 6. Quick Test (CLI)
If you don't want to run the API server, you can test direct text generation from the model via the Terminal:

**On Mac (MLX):**
```bash
uv run --with mlx-lm python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --adapter-path finetuning/adapters.safetensors \
    --prompt "<|im_start|>user\nComment obtenir un titre de séjour salarié?<|im_end|>\n<|im_start|>assistant\n<thinking>" \
    --max-tokens 500
```

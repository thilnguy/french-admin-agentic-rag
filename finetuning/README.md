# Hướng dẫn Fine-tune model cho French Admin Assistant

Đây là hướng dẫn thực hiện Phase 3 và Phase 4 của kế hoạch tối ưu chi phí LLM.

## 1. Chuẩn bị dữ liệu (Data Preparation)
Dữ liệu chuyên gia đã được chuẩn bị với 600 mẫu (bao gồm Reasoning CoT và data Clarification Injection):
`finetuning/data/train_expert_formatted.jsonl`


## 2. Phần cứng khuyến nghị
### Dành cho Linux (NVIDIA GPU)
- **GPU**: Khuyến nghị tối thiểu 24GB VRAM (RTX 3090, 4090, A10g, A100) để chạy QLoRA với batch size tốt.
- **RAM**: 32GB+.

### Dành cho Mac (Apple Silicon)
- **Chip**: M2/M3/M4 Max hoặc Ultra (Memory bandwidth cao).
- **Unified Memory**: Tối thiểu 32GB (Khuyến nghị 64GB+).

---

## 3. Cài đặt môi trường Training
Bạn nên sử dụng **Axolotl** hoặc **Unsloth** để đạt tốc độ cao nhất trên Linux, và **MLX** trên Mac.

### Linux (NVIDIA)
Khuyên dùng Docker chứa sẵn CUDA toolkit từ [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) hoặc dùng môi trường [Unsloth](https://github.com/unslothai/unsloth) để tốc độ training tăng x2 và giảm 50% VRAM:
```bash
# Cài đặt Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 4. Chạy Training
Dùng file cấu hình tôi đã chuẩn bị:
```bash
axolotl finetuning/axolotl_config.yml
```
Sau khi hoàn tất, model (LoRA adapters) sẽ được lưu tại thư mục `finetuning/qwen-7b-french-admin-distilled`.

---

## 5. Deploy và Sử dụng Model

Marianne AI hỗ trợ cả 2 hệ sinh thái Local lớn nhất hiện nay: vLLM (cho Linux/NVIDIA) và MLX (cho Mac/Apple Silicon).

### Lựa chọn A: Linux Server (Dùng vLLM)
vLLM là engine inference nhanh nhất hiện nay cho production server NVIDIA.

**Bước 1: Chạy Server vLLM**
```bash
# Cài đặt vLLM
pip install vllm

# Khởi chạy OpenAI-compatible server với model đã fine-tune (nhúng sẵn lora adapters)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules french_admin=finetuning/qwen-7b-french-admin-distilled \
    --port 8020
```

### Lựa chọn B: MacBook (Dùng MLX Server)
Đối với máy Mac, framework MLX của Apple là lựa chọn tối ưu 100%.

**Bước 1: Chạy Server MLX**
```bash
uv run --with mlx-lm python -m mlx_lm.server \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --adapter-path finetuning/adapters.safetensors \
    --port 8020
```

---

### Bước 2: Cấu hình Agent (Chung cho cả Linux & Mac)
Cập nhật file `.env` ở thư mục gốc để app trỏ request về Local Server thay vì gọi API Cloud:
```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:8020/v1
LOCAL_LLM_MODEL=french_admin # (hoặc tên model tuỳ bạn đặt theo MLX/vLLM)
```

### Bước 3: Kiểm tra chất lượng (Evaluation)
Để biết model mới xịn hơn GPT-4o thế nào ở môi trường Local của bạn, hãy chạy script đánh giá:
```bash
uv run python evals/llm_judge.py
```

---

## 6. Test nhanh (CLI)
Nếu không muốn chạy server API, bạn có thể test thử model generate chữ trực tiếp từ Terminal:

**Trên Mac (MLX):**
```bash
uv run --with mlx-lm python -m mlx_lm.generate \
    --model mlx-community/Qwen2.5-7B-Instruct-8bit \
    --adapter-path finetuning/adapters.safetensors \
    --prompt "<|im_start|>user\nComment obtenir un titre de séjour salarié?<|im_end|>\n<|im_start|>assistant\n<thinking>" \
    --max-tokens 500
```


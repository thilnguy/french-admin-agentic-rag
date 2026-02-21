# Hướng dẫn Fine-tune model cho French Admin Assistant

Đây là hướng dẫn thực hiện Phase 3 và Phase 4 của kế hoạch tối ưu chi phí LLM.

## 1. Chuẩn bị dữ liệu (Data Preparation)
Dữ liệu chuyên gia đã được chuẩn bị với 300 mẫu (bao gồm Reasoning CoT):
`finetuning/data/train_expert_formatted.jsonl`


## 2. Phần cứng khuyến nghị
- **GPU**: Cần ít nhất 24GB VRAM (NVIDIA RTX 3090, 4090, A10, A100) để chạy QLoRA.
- **RAM**: 32GB+.
- **Disk**: 50GB+ dung lượng trống.

## 3. Cài đặt môi trường Training
Bạn nên sử dụng **Axolotl** hoặc **Unsloth** để đạt tốc độ cao nhất.

```bash
# Cài đặt Axolotl (khuyên dùng Docker hoặc Conda)
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

## 4. Chạy Training
Dùng file cấu hình tôi đã chuẩn bị:
```bash
axolotl finetuning/axolotl_config.yml
```

Sau khi hoàn tất, model sẽ được lưu tại thư mục `finetuning/qwen-7b-french-admin-distilled`.

## 5. Deploy và Sử dụng Model (Mac M4)

Sau khi training xong, bạn có thể chạy model ngay lập tức bằng server OpenAI-compatible của MLX:

### Bước 1: Chạy Server
```bash
uv run --with mlx-lm python -m mlx_lm.server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter-path finetuning/adapters.safetensors \
    --port 8020
```

### Bước 2: Cấu hình Agent để sử dụng Local Model
Cập nhật file `.env` ở thư mục gốc:
```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:8020/v1
LOCAL_LLM_MODEL=qwen-7b-french-admin
```

### Bước 3: Kiểm tra chất lượng (Evaluation)
Để biết model mới xịn hơn GPT-4o thế nào, hãy chạy script đánh giá:
```bash
uv run python evals/llm_judge.py
```

## 6. Test nhanh (CLI)
Nếu không muốn chạy server, bạn có thể test thử một câu hỏi trực tiếp:
```bash
uv run --with mlx-lm python -m mlx_lm.generate \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter-path finetuning/adapters.safetensors \
    --prompt "<|im_start|>user\nComment obtenir un titre de séjour salarié?<|im_end|>\n<|im_start|>assistant\n<thinking>" \
    --max-tokens 500
```


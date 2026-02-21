# Hướng dẫn Fine-tune model cho French Admin Assistant

Đây là hướng dẫn thực hiện Phase 3 và Phase 4 của kế hoạch tối ưu chi phí LLM.

## 1. Chuẩn bị dữ liệu (Data Preparation)
Dữ liệu đã được chưng cất (distilled) từ GPT-4o và lưu tại:
`finetuning/data/train_samples.jsonl`

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

## 5. Deploy và Chuyển đổi Model
Sử dụng **vLLM** để host model cục bộ với OpenAI Compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen-7b-french-admin-distilled \
    --port 8000
```

Sau đó, hãy cập nhật file `.env` của project:
```env
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=qwen-7b-french-admin
```

Hệ thống sẽ tự động sử dụng model local mà không cần thay đổi code thêm.

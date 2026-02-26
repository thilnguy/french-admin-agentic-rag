import streamlit as st
import httpx
import json
import uuid

# --- Cấu hình giao diện Streamlit ---
st.set_page_config(page_title="Marianne AI - Hành chính Pháp", page_icon="🇫🇷", layout="centered")
st.title("🇫🇷 Marianne AI - Trợ lý Hành chính Pháp")
st.markdown("Hệ thống **Agentic RAG** hỗ trợ giải đáp luật và thủ tục cư trú Pháp (Tiếng Việt/Anh/Pháp).")

API_URL = "http://127.0.0.1:8001/chat/stream"

# Khởi tạo Session ID để giữ context (trí nhớ)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Xử lý khi người dùng nhập câu hỏi ---
if prompt := st.chat_input("Hỏi Marianne AI (VD: Mình bị mất thẻ cư trú, phải làm sao?)..."):
    # 1. Thêm câu hỏi vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý phản hồi từ Agent (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""

        # Gọi API backend FastAPI dạng streaming (Server-Sent Events)
        try:
            with httpx.stream("POST", API_URL, json={"query": prompt, "session_id": st.session_state.session_id}, timeout=30.0) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line.startswith("data:"):
                            try:
                                data = json.loads(line[5:])
                                event_type = data.get("type")
                                content = data.get("content", "")

                                if event_type == "status":
                                    status_placeholder.caption(f"🔄 {content}")
                                elif event_type == "token":
                                    full_response += content
                                    message_placeholder.markdown(full_response + "▌")
                                elif event_type == "error":
                                    st.error(f"Lỗi: {content}")
                            except json.JSONDecodeError:
                                pass
                else:
                    error_content = response.read().decode()
                    st.error(f"Lỗi kết nối API ({response.status_code}): {error_content}")
            
            # Xóa con trỏ nhấp nháy và lưu lịch sử
            message_placeholder.markdown(full_response)
            status_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except httpx.RequestError as e:
            st.error(f"Không thể kết nối đến Backend: {e}")
            st.info("💡 Bạn đã chạy `uv run uvicorn src.main:app --port 8001` chưa?")

# Sidebar tiện ích
with st.sidebar:
    st.header("⚙️ Tuỳ chọn")
    if st.button("🗑️ Xóa Context (New Chat)"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("---")
    st.markdown("**Session ID:**")
    st.code(st.session_state.session_id)
    st.caption("Dùng để debug logs trong backend.")

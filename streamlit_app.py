import streamlit as st
import httpx
import json
import uuid

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Marianne AI - French Administration", page_icon="🇫🇷", layout="centered")

st.markdown("""
<style>
/* 1. BLUE FOCUS BORDER FOR CHAT INPUT (Like ChatGPT) */
div[data-testid="stChatInputContainer"]:focus-within {
    border: 1px solid #0084ff !important;
    box-shadow: 0 0 0 1px #0084ff !important;
}

/* 2. RIGHT-ALIGN USER MESSAGES (Like ChatGPT) */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
    text-align: right;
}

/* Background styling for User messages */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
    background-color: #f0f2f6; 
    border-radius: 18px 18px 0px 18px;
    padding: 12px 16px;
    display: inline-block;
}

/* Background styling for Bot messages */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"] {
    background-color: white; 
    border-radius: 18px 18px 18px 0px;
    padding: 12px 16px;
    border: 1px solid #e0e0e0;
    display: inline-block;
}

/* Remove default styling for all Streamlit Chat Message blocks */
div[data-testid="stChatMessage"] {
    background-color: transparent !important;
}

/* Bottom padding to prevent overlap with chat input & footer */
.stApp {
    padding-bottom: 120px;
}

/* Footer disclaimer fixed at the bottom */
.footer-note {
    position: fixed;
    bottom: 5px;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 13px;
    color: #888;
    z-index: 100;
    padding: 0 10px;
    background-color: transparent;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🇫🇷 Marianne AI - French Administrative Assistant")
st.markdown("**Agentic RAG** system assisting with French residency laws and procedures (English/French/Vietnamese).")

API_URL = "http://127.0.0.1:8001/chat/stream"

# Initialize Session State
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}
if "session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.all_sessions[new_id] = {"title": "New Chat", "messages": [], "is_initial": True}

# Reference the current active messages
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.all_sessions[st.session_state.session_id]["messages"]

# Helper to switch sessions
def switch_session(s_id):
    st.session_state.session_id = s_id
    st.session_state.messages = st.session_state.all_sessions[s_id]["messages"]

# Display chat history for CURRENT session
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle user input ---
if prompt := st.chat_input("Ask Marianne AI (e.g., I lost my residence permit, what should I do?)..."):
    # 1. Add question to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update title if it's the first message
    current_session = st.session_state.all_sessions[st.session_state.session_id]
    if current_session["title"] == "New Chat":
        current_session["title"] = prompt[:25] + "..." if len(prompt) > 25 else prompt
        
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Handle Agent response (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""

        # Call FastAPI backend streaming API (Server-Sent Events)
        try:
            payload = {
                "query": prompt, 
                "session_id": st.session_state.session_id,
                "model": st.session_state.get("selected_model", "GPT-4o"),
            }
            with httpx.stream("POST", API_URL, json=payload, timeout=30.0) as response:
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
                                    st.error(f"Error: {content}")
                            except json.JSONDecodeError:
                                pass
                else:
                    error_content = response.read().decode()
                    st.error(f"API Connection Error ({response.status_code}): {error_content}")
            
            # Remove blinking cursor and save to history
            message_placeholder.markdown(full_response)
            status_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Sync to global session context
            st.session_state.all_sessions[st.session_state.session_id]["messages"] = st.session_state.messages

        except httpx.RequestError as e:
            st.error(f"Cannot connect to Backend: {e}")
            st.info("💡 Have you run `uv run uvicorn src.main:app --port 8001` ?")

# Disclaimer (Displayed persistently at the bottom via CSS)
st.markdown("<div class='footer-note'><em>Note: Information provided by Marianne AI is for reference only. For final decisions, consult service-public.fr.</em></div>", unsafe_allow_html=True)


# Sidebar utilities divided into sections
with st.sidebar:
    st.header("⚙️ Marianne AI Menu")
    
    with st.expander("🤖 Model Selection", expanded=True):
        st.selectbox("Model", ["GPT-4o", "Qwen Finetuned (Local)"], key="selected_model")
        st.caption("Change the AI brain driving Marianne.")
    
    with st.expander("📝 Chat History", expanded=True):
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.all_sessions[new_id] = {"title": "New Chat", "messages": [], "is_initial": False}
            switch_session(new_id)
            st.rerun()
            
        st.markdown("---")
        # Display sessions latest first
        for s_id, s_data in list(st.session_state.all_sessions.items())[::-1]:
            # Ẩn session mặc định ban đầu nếu nó chưa có tin nhắn nào
            if s_data.get("is_initial", False) and len(s_data["messages"]) == 0:
                continue
                
            # Highlight current session
            label = f"💬 {s_data['title']}"
            if st.button(label, key=s_id, use_container_width=True):
                switch_session(s_id)
                st.rerun()
            
    with st.expander("📊 Technical Info", expanded=False):
        st.markdown("**Session ID:**")
        st.code(st.session_state.session_id)
        st.caption("Used for debugging/tracing logs in the backend.")
        
    st.markdown("---")
    st.caption("v1.5 - Agentic RAG Architecture based on Qwen 7B & Qdrant")

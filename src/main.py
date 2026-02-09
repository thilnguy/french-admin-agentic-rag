import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from src.agents.orchestrator import AdminOrchestrator
from skills.polyglot_voice.main import speech_to_text, text_to_speech

load_dotenv()

app = FastAPI(title="French Administrative Agent API")
orchestrator = AdminOrchestrator()

@app.get("/")
def read_root():
    return {"status": "French Admin Agent is online"}

@app.post("/chat")
def chat(query: str, language: str = "fr"):
    """
    Standard text chat endpoint.
    """
    answer = orchestrator.handle_query(query, language)
    return {"answer": answer}

@app.post("/voice_chat")
async def voice_chat(audio: UploadFile = File(...), language: str = "fr"):
    """
    Multimodal endpoint: Speech -> Text -> Agent -> Answer -> Speech.
    """
    # 1. STT
    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio.read())
        
    user_text = speech_to_text(audio_path=temp_path, language=language)
    
    # 2. Agent Logic
    answer_text = orchestrator.handle_query(user_text, language)
    
    # 3. TTS
    audio_response_path = text_to_speech(text=answer_text, language=language)
    
    return {
        "user_text": user_text,
        "answer_text": answer_text,
        "audio_url": audio_response_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

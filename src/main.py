import uvicorn
import time
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.agents.orchestrator import AdminOrchestrator
from skills.polyglot_voice.main import speech_to_text, text_to_speech
from src.config import settings
from src.utils.logger import logger
from src.schemas import ChatRequest, ChatResponse, VoiceChatResponse

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AdminOrchestrator()

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please try again later."}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}

@app.get("/")
async def read_root():
    return {"status": "French Admin Agent is online", "mode": "Debug" if settings.DEBUG else "Production"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Standard text chat endpoint.
    """
    logger.info(f"Received chat request: {request.query} [{request.language}]")
    try:
        answer = await orchestrator.handle_query(request.query, request.language, request.session_id)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_chat", response_model=VoiceChatResponse)
async def voice_chat(audio: UploadFile = File(...), language: str = "fr", session_id: str = "default"):
    """
    Multimodal endpoint: Speech -> Text -> Agent -> Answer -> Speech.
    """
    logger.info(f"Received voice chat request [{language}]")
    try:
        # 1. STT
        temp_path = f"temp_{audio.filename}"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())
            
        user_text = speech_to_text(audio_path=temp_path, language=language)
        logger.info(f"Transcribed audio: {user_text}")
        
        # 2. Agent Logic
        answer_text = await orchestrator.handle_query(user_text, language, session_id)
        
        # 3. TTS
        audio_response_path = text_to_speech(text=answer_text, language=language)
        
        return VoiceChatResponse(
            user_text=user_text,
            answer_text=answer_text,
            audio_url=audio_response_path
        )
    except Exception as e:
        logger.error(f"Error in voice_chat endpoint: {e}")
        # Clean up temp file if needed
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

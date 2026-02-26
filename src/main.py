import os
import tempfile
import uvicorn
import time
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request,
    HTTPException,
    Security,
    Depends,
    Query,
)
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.agents.orchestrator import AdminOrchestrator
from skills.polyglot_voice.main import speech_to_text, text_to_speech
from src.config import settings
from src.utils.logger import logger
from src.schemas import ChatRequest, ChatResponse, VoiceChatResponse, FeedbackRequest
from skills.legal_retriever.main import warmup as warmup_retriever
from prometheus_fastapi_instrumentator import Instrumentator
from src.utils import metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — warming Qdrant client and embeddings model...")
    try:
        warmup_retriever()
    except Exception as e:
        logger.warning(f"Warmup failed (services may not be ready): {e}")
    logger.info("French Admin Agent ready.")
    yield
    logger.info("Shutting down — closing connections...")
    try:
        await orchestrator.cache.aclose()
    except Exception:
        pass


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# Instrument the app (exposes /metrics)
Instrumentator().instrument(app).expose(app)

# OpenTelemetry Instrumentation
if settings.OTEL_ENABLED:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware — restrict to configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AdminOrchestrator()


# Global Exception Handler
import openai

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {str(exc)}", exc_info=True)
    
    # Catch specific API errors from LLM providers
    if isinstance(exc, openai.RateLimitError):
        return JSONResponse(
            status_code=429,
            content={"detail": "Le service est surchargé (Rate Limit OpenAI). Veuillez réessayer dans quelques minutes."},
        )
    elif isinstance(exc, openai.APIConnectionError):
        return JSONResponse(
            status_code=503,
            content={"detail": "Erreur de connexion avec le service AI. Veuillez vérifier l'état du réseau."},
        )
    elif isinstance(exc, openai.AuthenticationError):
        return JSONResponse(
            status_code=401,
            content={"detail": "Erreur d'authentification API. Veuillez vérifier votre clé API OpenAI."},
        )
        
    return JSONResponse(
        status_code=500,
        content={"detail": "Une erreur interne s'est produite. Veuillez réessayer plus tard."},
    )


# Security Dependency
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = None,
):
    if settings.API_KEY:
        # Check header first, then query param
        key = api_key_header or api_key_query
        if key == settings.API_KEY:
            return key
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s"
    )
    return response


@app.get("/health")
async def health_check():
    """Deep health check — verifies Redis and Qdrant connectivity."""
    import redis
    from qdrant_client import QdrantClient

    # Check Redis
    redis_ok = False
    try:
        r = redis.from_url(settings.REDIS_URL)
        redis_ok = r.ping()
    except Exception:
        pass

    # Check Qdrant
    qdrant_ok = False
    try:
        q = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=False,
            timeout=3,
        )
        q.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.error(f"Health check Qdrant failed: {e}")
        pass

    status = "healthy" if (redis_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "version": settings.APP_VERSION,
        "dependencies": {"redis": redis_ok, "qdrant": qdrant_ok},
    }


@app.get("/")
async def read_root():
    return {
        "status": "French Admin Agent is online",
        "mode": "Debug" if settings.DEBUG else "Production",
    }


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
@limiter.limit(settings.RATE_LIMIT)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Standard text chat endpoint.
    """
    import asyncio

    logger.info(
        f"Received chat request: {chat_request.query} [{chat_request.language}]"
    )
    try:
        answer = await asyncio.wait_for(
            orchestrator.handle_query(
                chat_request.query, chat_request.language, chat_request.session_id
            ),
            timeout=60.0,  # 60s max — prevents 'Failed to fetch' on complex queries
        )
        return ChatResponse(answer=answer)
    except asyncio.TimeoutError:
        logger.error(f"Query timed out after 60s: {chat_request.query}")
        timeout_messages = {
            "fr": "Désolé, la requête a pris trop de temps. Veuillez réessayer avec une question plus courte.",
            "en": "Sorry, the request timed out. Please try again with a shorter question.",
            "vi": "Xin lỗi, yêu cầu mất quá nhiều thời gian. Vui lòng thử lại với câu hỏi ngắn hơn.",
        }
        lang_key = chat_request.language.lower()[:2]
        msg = timeout_messages.get(lang_key, timeout_messages["fr"])
        return ChatResponse(answer=msg)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream", dependencies=[Depends(get_api_key)])
@limiter.limit(settings.RATE_LIMIT)
async def chat_stream(
    request: Request,
    query: str = Query(
        ..., min_length=1, max_length=500, description="User query (1-500 chars)"
    ),
    language: str = Query(
        "fr", pattern="^(fr|en|vi)$", description="Response language: fr, en, or vi"
    ),
    session_id: str = Query(
        "default", min_length=1, max_length=100, description="Session identifier"
    ),
    api_key: str = None,  # Captured by Depends(get_api_key) but needed in signature for OpenAPI
):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Yields JSON events: {"type": "token"|"status"|"error", "content": "..."}
    """
    logger.info(f"Received stream request: {query} [{language}]")

    async def event_generator():
        try:
            async for event in orchestrator.stream_query(query, language, session_id):
                # SSE format: data: <json>\n\n
                import json

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f'data: {{"type": "error", "content": "{str(e)}"}}\n\n'
        finally:
            yield "data: [DONE]\n\n"

    from fastapi.responses import StreamingResponse

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post(
    "/voice_chat", response_model=VoiceChatResponse, dependencies=[Depends(get_api_key)]
)
@limiter.limit(settings.RATE_LIMIT)
async def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    language: str = "fr",
    session_id: str = "default",
):
    """
    Multimodal endpoint: Speech -> Text -> Agent -> Answer -> Speech.
    """
    logger.info(f"Received voice chat request [{language}]")
    temp_path = None
    try:
        # 1. STT — Use secure temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await audio.read())
            temp_path = tmp.name

        user_text = speech_to_text(audio_path=temp_path, language=language)
        logger.info(f"Transcribed audio: {user_text}")

        # 2. Agent Logic
        answer_text = await orchestrator.handle_query(user_text, language, session_id)

        # 3. TTS
        audio_response_path = text_to_speech(text=answer_text, language=language)

        return VoiceChatResponse(
            user_text=user_text, answer_text=answer_text, audio_url=audio_response_path
        )
    except Exception as e:
        logger.error(f"Error in voice_chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback for a chat interaction.
    """
    # Log to Prometheus metrics
    metrics.USER_FEEDBACK.labels(score=feedback.score).inc()

    logger.info(
        f"Received feedback: {feedback.score} for session {feedback.session_id}"
    )
    return {"status": "received", "score": feedback.score}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

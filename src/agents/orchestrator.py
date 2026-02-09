import os
import redis
import json
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from skills.legal_retriever.main import retrieve_legal_info
from skills.admin_translator import translate_admin_text

class AdminOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.retriever = retrieve_legal_info
        self.translator = translate_admin_text
        
        # Initialize Redis Cache
        self.cache = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )

    def handle_query(self, query: str, user_lang: str = "fr"):
        """
        Main orchestration logic with Guardrails, Caching, and Query Translation.
        """
        # Cache Key based on query and language
        cache_key = f"agent_res:{hashlib.md5((query + user_lang).encode()).hexdigest()}"
        
        # Bypass cache if DEBUG_RAG=true
        if os.getenv("DEBUG_RAG", "false").lower() != "true":
            cached_res = self.cache.get(cache_key)
            if cached_res:
                return cached_res

        from src.shared.guardrails import guardrail_manager
        
        # Guardrail 1: Topic Validation
        is_valid, reason = guardrail_manager.validate_topic(query)
        if not is_valid:
            return f"Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Lý do: {reason}"

        # OPTIMIZATION: Translate query to French for better retrieval accuracy
        retrieval_query = query
        if user_lang.lower() != "fr":
            print(f"DEBUG: Translating query to French for retrieval...")
            retrieval_query = self.translator(text=query, target_language="French")
            print(f"DEBUG: Retrieval query (FR): {retrieval_query}")

        # Step 1: Search for info (RAG)
        context = self.retriever(query=retrieval_query)
        context_text = "\n".join([f"Source {d['source']}: {d['content']}" for d in context])
        
        # Step 2: Formulate answer
        system_prompt = """You are a French Administrative Expert. 
        Your task is to answer the user's question accurately based ONLY on the provided context.
        Write your internal answer strictly in French. Do not include any meta-talk or apologies about language capabilities."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context_text}\n\nQuestion in {user_lang}: {query}")
        ]
        
        french_answer = self.llm.invoke(messages).content
        
        # Guardrail 2: Hallucination Check
        if not guardrail_manager.check_hallucination(context_text, french_answer):
            french_answer = "Désolé, tôi không tìm thấy thông tin chính xác trong cơ sở dữ liệu để trả lời câu hỏi này một cách an toàn."

        # Step 3: Polyglot Translation
        final_answer = french_answer
        if user_lang.lower() in ["english", "en"]:
            final_answer = self.translator(text=french_answer, target_language="English")
        elif user_lang.lower() in ["vietnamese", "vi"]:
            final_answer = self.translator(text=french_answer, target_language="Vietnamese")
            
        # Guardrail 3: Add Disclaimer
        final_response = guardrail_manager.add_disclaimer(final_answer, user_lang)
        
        # Save to cache (TTL 1 hour)
        try:
            self.cache.setex(cache_key, 3600, final_response)
        except:
            pass
            
        return final_response

def run_agent(input_data: dict):
    orchestrator = AdminOrchestrator()
    return orchestrator.handle_query(
        query=input_data.get("query"),
        user_lang=input_data.get("language", "fr")
    )

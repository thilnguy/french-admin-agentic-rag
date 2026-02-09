import antigravity as ag
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class AdminOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        # Register skills
        self.retriever = ag.get_skill("legal_retriever")
        self.translator = ag.get_skill("admin_translator")

    def handle_query(self, query: str, user_lang: str = "fr"):
        """
        Main orchestration logic with Guardrails.
        """
        from src.shared.guardrails import guardrail_manager
        
        # Guardrail 1: Topic Validation
        is_valid, reason = guardrail_manager.validate_topic(query)
        if not is_valid:
            return f"Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Lý do: {reason}"

        # Step 1: Search for info (RAG)
        context = self.retriever(query=query)
        context_text = "\n".join([f"Source {d['source']}: {d['content']}" for d in context])
        
        # Step 2: Formulate answer
        system_prompt = """You are a French Administrative Assistant. 
        Answer the user's question accurately based ONLY on the provided context.
        Provide the answer in French initially."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context_text}\n\nQuestion: {query}")
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
        return guardrail_manager.add_disclaimer(final_answer, user_lang)

# For Antigravity Graph Mode (if used)
@ag.agent(name="french_admin_agent")
def run_agent(input_data: dict):
    orchestrator = AdminOrchestrator()
    return orchestrator.handle_query(
        query=input_data.get("query"),
        user_lang=input_data.get("language", "fr")
    )

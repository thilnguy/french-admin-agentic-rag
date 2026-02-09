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
        Main orchestration logic.
        1. Understand intent.
        2. Retrieve legal info (always in French for accuracy).
        3. Translate back to user language if needed.
        """
        # Step 1: Search for info (RAG)
        context = self.retriever(query=query)
        
        # Step 2: Formulate answer
        system_prompt = """You are a French Administrative Assistant. 
        Answer the user's question accurately based ONLY on the provided context.
        Provide the answer in French initially."""
        
        context_text = "\n".join([f"Source {d['source']}: {d['content']}" for d in context])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context_text}\n\nQuestion: {query}")
        ]
        
        french_answer = self.llm.invoke(messages).content
        
        # Step 3: Polyglot Translation
        if user_lang.lower() in ["english", "en"]:
            return self.translator(text=french_answer, target_language="English")
        elif user_lang.lower() in ["vietnamese", "vi"]:
            return self.translator(text=french_answer, target_language="Vietnamese")
            
        return french_answer

# For Antigravity Graph Mode (if used)
@ag.agent(name="french_admin_agent")
def run_agent(input_data: dict):
    orchestrator = AdminOrchestrator()
    return orchestrator.handle_query(
        query=input_data.get("query"),
        user_lang=input_data.get("language", "fr")
    )

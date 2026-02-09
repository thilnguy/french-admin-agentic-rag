import antigravity as ag
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@ag.skill(name="admin_translator")
def translate_admin_text(text: str, target_language: str):
    """
    Translates French administrative text into English or Vietnamese, 
    ensuring technical terms (e.g., Prefecture, Titre de séjour) are correctly contextually translated.
    target_language: 'English' or 'Vietnamese'
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert translator specializing in French Administrative Law. 
        Translate the following text into {target_language}.
        IMPORTANT: 
        - Maintain the legal accuracy of terms.
        - If a term has no direct equivalent, keep the French term in parentheses after your translation.
        - For Vietnamese, ensure the tone is formal and appropriate for administrative guidance."""),
        ("user", "{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text, "target_language": target_language})

if __name__ == "__main__":
    # Example: print(translate_admin_text("Demande de titre de séjour à la préfecture", "Vietnamese"))
    pass

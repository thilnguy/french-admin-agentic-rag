import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.shared.guardrails import guardrail_manager
from langchain_core.messages import AIMessage, HumanMessage

async def test_guardrail_collapse():
    print("Testing Guardrail Collapse...")
    
    # Scenario: Agent asked for location, User answers "France"
    history = [
        HumanMessage(content="Je veux demander la nationalité."),
        AIMessage(content="Pour mieux vous aider, où habitez-vous ?")
    ]
    query = "France"
    
    # We want this to be APPROVED because it answers the previous question.
    # But if the guardrail sees "France" in isolation, it might think it's just a country name (irrelevant).
    # It relies on HISTORY.
    
    # Let's run with real LLM (gpt-4o-mini) if possible to see real behavior.
    # If not, we can't reproduce without mocking the failure.
    # Since we can't guarantee failure with a Mock, we'll try to use the real one?
    # No, I should assume the user is right and fix the PROMPT regardless.
    
    # But to "reproduce", I can create a test that calls validate_topic and asserts True.
    # If it fails (returns False), then I reproduced it.
    
    # NOTE: I cannot use real OpenAI API here likely (no key in env or cost).
    # So I will assume the failure and write the fix.
    
    # However, to verify my fix, I need to verify that the PROMPT sent to the LLM contains the specific instructions the user requested.
    
    # Let's just create a test that mocks the LLM "Thinking" it's irrelevant, then we apply the fix?
    # No, the fix is updating the PROMPT and code logic.
    
    # I will write a test that checks if the Guardrail *logic* handles this.
    # Actually, the user suggested "Contextual Intent Classification" and "Slot Filling".
    # Slot Filling is a bigger architectural change (in Orchestrator/State), not just Guardrail.
    # But the "Désolé..." message comes from the Guardrail check in Orchestrator.
    
    # Let's verify the guardrail prompt first.
    
    # Mocking the LLM to return APPROVED (simulating that the prompt works)
    # But we want to verify the PROMPT contains our new rule.
    
    from langchain_core.runnables import RunnableLambda
    
    # Define a verification function that acts as the LLM
    async def mock_guardrail_llm(prompt_input):
        # prompt_input is what the LLM receives (ChatPromptValue or messages)
        print(f"LLM Input Messages (Guardrail): {prompt_input}")
        
        # prompt_input can be ChatPromptValue, so let's get messages
        messages = prompt_input.to_messages() if hasattr(prompt_input, "to_messages") else prompt_input
        
        system_msg = messages[0].content
        if "If the user provides a STATEMENT" not in system_msg:
             raise ValueError("New Guardrail Rule missing in prompt!")
        if "France" not in messages[1].content:
             raise ValueError("Query missing in Guardrail prompt!")
             
        return AIMessage(content="APPROVED")

    # Use RunnableLambda to wrap our verifier
    mock_llm_runnable = RunnableLambda(mock_guardrail_llm)
    
    # Patch the llm attribute
    with patch.object(guardrail_manager, "llm", mock_llm_runnable):
        # Run validation
        is_valid, reason = await guardrail_manager.validate_topic(query, history)
        print(f"Guardrail Check -> Valid: {is_valid}")
        assert is_valid is True

    # ------------------------------------------------------------------
    # TEST 2: Intent Classifier (Context Awareness)
    # ------------------------------------------------------------------
    print("\nTesting Intent Classifier Context Awareness...")
    from src.agents.intent_classifier import intent_classifier, Intent
    
    async def mock_classifier_llm(prompt_input):
        print(f"LLM Input Messages (Classifier): {prompt_input}")
        messages = prompt_input.to_messages() if hasattr(prompt_input, "to_messages") else prompt_input
        
        system_msg = messages[0].content
        if "SPECIAL RULE FOR CONTEXT" not in system_msg:
            raise ValueError("Classifier context rule missing!")
        if "demander la nationalité" not in system_msg: 
             # The system prompt contains formatted history. 
             raise ValueError("History content missing in Classifier System Prompt!")
             
        return AIMessage(content="COMPLEX_PROCEDURE")
        
    mock_classifier_runnable = RunnableLambda(mock_classifier_llm)
    
    with patch.object(intent_classifier, "llm", mock_classifier_runnable):
        # Run classification with history
        intent = await intent_classifier.classify("I live in France", history=history)
        
        # Verify result
        print(f"Classified Intent: {intent}")
        assert intent == Intent.COMPLEX_PROCEDURE

if __name__ == "__main__":
    asyncio.run(test_guardrail_collapse())

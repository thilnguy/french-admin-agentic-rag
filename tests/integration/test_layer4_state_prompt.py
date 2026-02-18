import asyncio
from unittest.mock import patch
from src.agents.procedure_agent import procedure_agent
from src.agents.state import AgentState, UserProfile


async def test_layer4_state_prompt():
    print("Testing Layer 4: State-Based System Prompt...")

    # Mock Retrieve to return a generic doc about Visa
    mock_docs = [
        {
            "content": "Pour un visa long séjour, les pièces dépendent de votre nationalité. Si vous êtes Américain, vous devez fournir X. Si vous êtes Chinois, Y.",
            "source": "service-public.fr",
        }
    ]

    # Case 1: Profile has Nationality -> Agent should NOT ask, but ANSWER specific to American.
    print("\nCase 1: Profile HAS Nationality (Américaine)")
    state_with_profile = AgentState(
        session_id="test_layer4", user_profile=UserProfile(nationality="Américaine")
    )

    # We mock _run_chain to intercept the input and verify 'user_profile' is passed
    # But we also want to see the output.
    # Let's actually run the LLM (or mock it if we want save cost/time).
    # For robust verification of prompt logic, we should probably mock LLM but verify the PROMPT input.
    # However, user wants verification of "functionality".
    # Let's run it with a mock LLM that returns a string proving it saw the profile?
    # Or just trust the code change?

    # Let's mock the LLM response to simulate "I see you are American..."
    # But better: Mock the chain invocation and check the 'user_profile' in input_data.

    with patch.object(
        procedure_agent, "_run_chain", side_effect=procedure_agent._run_chain
    ) as mock_run_chain:
        # actually running the chain might require real OpenAI key. settings.OPENAI_API_KEY is mocked in conftest?
        # If conftest mocks it, real network calls will fail or we need to mock LLM.
        # Let's mock the LLM itself to avoid network calls.

        with patch.object(procedure_agent, "llm") as mock_llm:
            mock_llm.return_value = "Response likely using profile"  # Fake response
            # We fail because _run_chain expects an Awaitable from chain.ainvoke?
            # Creating a mock chain is complex.

            # Let's just check if _explain_procedure calls _run_chain with correct dict.
            pass

    # Real implementation of the test:
    # We invoke _explain_procedure directly.
    query = "Quels documents pour mon visa ?"

    with patch.object(procedure_agent, "_run_chain") as mock_run_chain:
        mock_run_chain.return_value = "Mocked Response"

        await procedure_agent._explain_procedure(
            query, mock_docs, state_with_profile.user_profile.model_dump()
        )

        # Verify call args
        args, kwargs = mock_run_chain.call_args
        # args[0] is the chain, args[1] is input_data
        input_data = (
            args[1] if len(args) > 1 else kwargs.get("input_data")
        )  # wait, _run_chain signature is (chain, input_data)

        # Check if input_data contains our profile
        assert "user_profile" in input_data, "user_profile missing from input_data"
        assert (
            input_data["user_profile"]["nationality"] == "Américaine"
        ), "Profile nationality mismatch"

        print("SUCCESS: user_profile was passed to the prompt chain.")


if __name__ == "__main__":
    asyncio.run(test_layer4_state_prompt())

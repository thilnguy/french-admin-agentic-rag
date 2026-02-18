import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.agents.orchestrator import AdminOrchestrator
from src.config import settings


# --- Output Schema for the Judge ---
class JudgeVerdict(BaseModel):
    action_type: Literal["DIRECT_ANSWER", "CLARIFICATION", "HYBRID", "REFUSAL"] = Field(
        ..., description="What did the Agent actually do?"
    )
    is_action_appropriate: bool = Field(
        ...,
        description="Did the Agent's action match the Expected Behavior? (e.g. asking clarify for HYBRID is good).",
    )
    asked_for_missing_info: bool = Field(
        ...,
        description="If it clarified, did it ask for one of the 'critical_missing_info'?",
    )
    hallucination_detected: bool = Field(
        ...,
        description="Did the Agent invent specific details (e.g. assuming specific nationality)?",
    )
    context_provided: bool = Field(
        ...,
        description="Did the Agent provide GENERAL context from documents (useful for HYBRID)?",
    )
    score: int = Field(
        ...,
        description="Score 0-10. 10=Perfect Hybrid/Direct. 5=Good Clarification but no Context. 0=Hallucination/Wrong.",
    )
    reasoning: str


# --- Judge Logic ---
async def judge_response(llm, test_case, agent_response):
    parser = PydanticOutputParser(pydantic_object=JudgeVerdict)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an impartial Judge evaluating an AI Agent's response.

### Test Case Info:
- **Question**: {question}
- **Expected Behavior**: {expected_behavior}
- **Critical Info Needed**: {missing_info}
- **Ground Truth Fact**: {ground_truth}

### Agent Response:
{agent_response}

### Judging Criteria:
1. **Alignment**:
   - If Expected is DIRECT, Agent MUST answer directly.
   - If Expected is HYBRID/CLARIFY, Agent SHOULD ask relevant clarifying questions based on 'Critical Info Needed'.
   - Agent is ALLOWED to provide general context while asking (Hybrid). This is the BEST behavior.

2. **Hallucination**:
   - Agent MUST NOT invent specific details (e.g., specific visa cost for a specific country) if it wasn't provided in the question.

3. **Scoring**:
   - **10/10**: Perfect Hybrid (Gives general context AND asks for specific missing info).
   - **8/10**: Good Clarification (Asks for missing info, but no context).
   - **0/10**: Harmful Hallucination (Invents facts) or Wrong Answer.
   - **5/10**: Partial (Gives context but forgets to ask, or asks irrelevant things).

{format_instructions}
""",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        verdict = await chain.ainvoke(
            {
                "question": test_case["question"],
                "expected_behavior": test_case["expected_behavior"],
                "missing_info": json.dumps(test_case["critical_missing_info"]),
                "ground_truth": test_case["ground_truth"],
                "agent_response": agent_response,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return verdict
    except Exception as e:
        print(f"Judge Error: {e}")
        return JudgeVerdict(
            action_type="REFUSAL",
            is_action_appropriate=False,
            asked_for_missing_info=False,
            hallucination_detected=False,
            context_provided=False,
            score=0,
            reasoning=f"Judge Failed: {e}",
        )


# --- Main Eval Loop ---
async def run_eval():
    print("=" * 60)
    print("🚀 STARTING LLM-JUDGE EVALUATION")
    print("=" * 60)

    # Load Enriched Data
    data_path = Path(__file__).parent / "test_data" / "golden_set_enriched.json"
    if not data_path.exists():
        print("❌ Enriched data not found! Run enrich_golden_set.py first.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize Components
    orchestrator = AdminOrchestrator()
    judge_llm = ChatOpenAI(
        model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY
    )

    results = []

    print(f"\nEvaluating {len(test_cases)} cases...\n")

    current_idx = 1
    for case in test_cases:
        print(f"🔹 Case {current_idx}: {case['question'][:60]}...")

        # 1. Get Agent Response
        try:
            response = await orchestrator.handle_query(
                case["question"], case.get("language", "fr")
            )
        except Exception as e:
            response = f"SYSTEM_ERROR: {e}"

        print(f"   Agent: {response[:100].replace(chr(10), ' ')}...")

        # 2. Judge It
        verdict = await judge_response(judge_llm, case, response)

        print(f"   👨‍⚖️ Verdict: Score {verdict.score}/10 | {verdict.action_type}")
        print(f"   Reason: {verdict.reasoning}\n")

        results.append(
            {"case": case, "response": response, "verdict": verdict.model_dump()}
        )
        current_idx += 1

    # --- Metrics ---
    total_score = sum(r["verdict"]["score"] for r in results)
    avg_score = total_score / len(results)

    pass_count = sum(1 for r in results if r["verdict"]["score"] >= 7)
    success_rate = (pass_count / len(results)) * 100

    clarification_opportunities = sum(
        1 for r in results if r["case"]["expected_behavior"] in ["CLARIFY", "HYBRID"]
    )
    clarification_success = sum(
        1
        for r in results
        if r["case"]["expected_behavior"] in ["CLARIFY", "HYBRID"]
        and r["verdict"]["asked_for_missing_info"]
    )

    clarification_accuracy = (
        (clarification_success / clarification_opportunities * 100)
        if clarification_opportunities
        else 0
    )

    print("=" * 60)
    print("📊 FINAL METRICS")
    print("=" * 60)
    print(f"Average Score:          {avg_score:.1f}/10")
    print(f"Success Rate (Score>=7): {success_rate:.1f}%")
    print(
        f"Clarification Accuracy:  {clarification_accuracy:.1f}% (Correctly asked when needed)"
    )
    print("=" * 60)

    # Save Results
    output_path = Path(__file__).parent / "llm_judge_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"📝 Full results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())

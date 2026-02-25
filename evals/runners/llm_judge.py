import asyncio
import argparse
import json
import redis
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from unittest.mock import patch, AsyncMock, MagicMock
from src.agents.orchestrator import AdminOrchestrator
from src.config import settings

# Rate limiting config
EVAL_DELAY_BETWEEN_CASES = 3.0  # seconds between test cases
EVAL_DELAY_WITHIN_CASE = 2.0  # seconds between agent call and judge call
EVAL_MAX_RETRIES = 3  # max retries on 429
EVAL_RETRY_BASE_DELAY = 2.0  # base delay for exponential backoff


async def with_rate_limit_retry(coro_fn, *args, **kwargs):
    """Calls an async function with exponential backoff on 429 errors."""
    for attempt in range(EVAL_MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                wait = EVAL_RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"   ⏳ Rate limit hit (attempt {attempt+1}/{EVAL_MAX_RETRIES}). Waiting {wait:.0f}s..."
                )
                await asyncio.sleep(wait)
                if attempt == EVAL_MAX_RETRIES - 1:
                    raise  # Re-raise on last attempt
            else:
                raise  # Non-rate-limit errors bubble up immediately


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
async def run_eval(data_file: str = None, limit: int = None, output_file: str = None):
    print("=" * 60)
    print("🚀 STARTING LLM-JUDGE EVALUATION")
    print("=" * 60)

    # CRITICAL: Flush Redis before eval to prevent stale state from contaminating results.
    # We flush TWO key namespaces:
    # 1. agent_res:*   — cached final responses (prevents serving old answers)
    # 2. agent_state:* — session state (prevents GoalExtractor reading stale core_goal
    #                    from a previous test case's session)
    # NOTE: GoalExtractor/QueryRewriter are KEPT — they are essential for production
    # multi-turn conversations. The issue is eval isolation, not the components themselves.
    try:
        r = redis.Redis(host="localhost", port=6379, db=0)
        res_keys = r.keys("agent_res:*") or []
        state_keys = r.keys("agent_state:eval_case_*") or []
        all_keys = res_keys + state_keys
        if all_keys:
            r.delete(*all_keys)
        print(
            f"🧹 Flushed {len(res_keys)} cached responses + {len(state_keys)} eval states from Redis."
        )
    except Exception as e:
        print(f"⚠️  Could not flush Redis cache: {e} (continuing anyway)")

    # Load Enriched Data
    if data_file:
        data_path = Path(data_file)
    else:
        data_path = Path(__file__).parent.parent / "data" / "benchmarks" / "ds_eval_blind_9.8_gpt_4o.json"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if limit:
        test_cases = test_cases[:limit]

    # Initialize Components
    print(f"📡 Model Provider: {settings.LLM_PROVIDER}")
    print(f"🤖 Local Model ID: {settings.LOCAL_LLM_MODEL}")
    if settings.LLM_PROVIDER == "openai":
        print("⚠️  WARNING: Evaluating OpenAI model, NOT the local fine-tuned model!")
    else:
        print("✅ SUCCESS: Evaluating local model configuration.")

    judge_llm = ChatOpenAI(
        model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY
    )

    results = []

    print(f"\nEvaluating {len(test_cases)} cases...\n")

    current_idx = 1
    for case in test_cases:
        print(f"🔹 Case {current_idx}: {case['question'][:60]}...")

        # 1. Get Agent Response
        # CRITICAL: Use unique session_id per test case to prevent state contamination.
        # GoalExtractor/QueryRewriter are KEPT for production multi-turn support.
        # We isolate eval by: (a) unique session_id, (b) deleting the session state before each case.
        test_session_id = f"eval_case_{current_idx}"

        # Delete this session's state so GoalExtractor starts fresh (no stale core_goal)
        try:
            r.delete(f"agent_state:{test_session_id}")
        except Exception:
            pass

        # --- Agent Call (with retry on 429) ---
        try:
            from src.agents.state import AgentState
            
            with (
                patch("src.agents.orchestrator.redis.Redis", new_callable=MagicMock),
                patch("src.memory.manager.redis.from_url", new_callable=MagicMock),
                patch("src.agents.orchestrator.memory_manager") as mock_mem,
                patch("skills.legal_retriever.main.retrieve_legal_info", new_callable=AsyncMock) as mock_retriever,
                patch("src.agents.orchestrator.retrieve_legal_info", new_callable=AsyncMock) as mock_retriever_orch,
                patch("src.agents.procedure_agent.retrieve_legal_info", new_callable=AsyncMock) as mock_retriever_proc,
                patch("src.agents.legal_agent.retrieve_legal_info", new_callable=AsyncMock) as mock_retriever_legal,
            ):
                # Initialize Orchestrator INSIDE the patch so it picks up mocked Redis/Memory
                orchestrator = AdminOrchestrator()

                # Mock memory: Ensure a FRESH state object is returned for EACH call
                async def mock_load_state(sid):
                    return AgentState(session_id=sid, messages=[])

                mock_mem.load_agent_state = AsyncMock(side_effect=mock_load_state)
                mock_mem.save_agent_state = AsyncMock()
                
                # Mock retrieval: return the ground truth as the "document"
                mock_docs = [{"content": case["ground_truth"], "source": "golden_dataset"}]
                mock_retriever.return_value = mock_docs
                mock_retriever_orch.return_value = mock_docs
                mock_retriever_proc.return_value = mock_docs
                mock_retriever_legal.return_value = mock_docs
                
                # Ensure orchestrator's explicit cache handles gracefully
                orchestrator.cache = AsyncMock()
                orchestrator.cache.get.return_value = None
                
                response = await with_rate_limit_retry(
                    orchestrator.handle_query,
                    case["question"],
                    case.get("language", "fr"),
                    session_id=test_session_id,
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            response = f"SYSTEM_ERROR: {e}"

        print(f"   Agent: {response[:100].replace(chr(10), ' ')}...")


        # Brief pause between agent call and judge call
        await asyncio.sleep(EVAL_DELAY_WITHIN_CASE)

        # --- Judge Call (with retry on 429) ---
        try:
            verdict = await with_rate_limit_retry(
                judge_response, judge_llm, case, response
            )
        except Exception as e:
            print(f"   Judge Error (after retries): {e}")
            verdict = JudgeVerdict(
                action_type="REFUSAL",
                is_action_appropriate=False,
                asked_for_missing_info=False,
                hallucination_detected=False,
                context_provided=False,
                score=0,
                reasoning=f"Judge Failed after retries: {e}",
            )

        print(f"   👨‍⚖️ Verdict: Score {verdict.score}/10 | {verdict.action_type}")
        print(f"   Reason: {verdict.reasoning}\n")

        results.append(
            {"case": case, "response": response, "verdict": verdict.model_dump()}
        )
        current_idx += 1

        # Delay between cases to respect OpenAI rate limits
        if current_idx <= len(test_cases):
            print(f"   ⏱️  Waiting {EVAL_DELAY_BETWEEN_CASES:.0f}s before next case...")
            await asyncio.sleep(EVAL_DELAY_BETWEEN_CASES)

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
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(__file__).parent.parent / "results" / "llm_judge_results_qwen2.5_7B_8bit.json"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"📝 Full results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Judge Evaluation")
    parser.add_argument("--data", type=str, help="Path to test data JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--output", type=str, help="Path to output results JSON file")
    args = parser.parse_args()

    asyncio.run(run_eval(data_file=args.data, limit=args.limit, output_file=args.output))

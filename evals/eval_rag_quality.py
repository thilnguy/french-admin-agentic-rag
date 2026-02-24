"""
RAG Quality Evaluation using RAGChecker.

RAGChecker (Amazon Science) provides fine-grained, claim-level evaluation.
Uses OpenAI gpt-4o-mini as extractor/checker (cheaper than Ragas).

Metrics:
- Overall: precision, recall, f1
- Retriever: claim_recall, context_precision
- Generator: faithfulness, hallucination, context_utilization,
             noise_sensitivity, self_knowledge
"""

import asyncio
import json
from pathlib import Path
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from src.agents.orchestrator import AdminOrchestrator
from skills.legal_retriever.main import retrieve_legal_info
from src.config import settings


async def collect_rag_results(test_cases):
    """Run RAG pipeline and format results for RAGChecker."""
    orchestrator = AdminOrchestrator()
    results = []

    for i, case in enumerate(test_cases):
        question = case["question"]
        lang = case.get("language", "fr")
        ground_truth = case["ground_truth"]

        print(f"  [{i+1}/{len(test_cases)}] {question[:60]}...")

        # Retrieve context
        retrieved_docs = await retrieve_legal_info(question, domain="general")
        context_list = []
        for j, doc in enumerate(retrieved_docs or []):
            context_list.append({"doc_id": f"doc_{j}", "text": doc["content"]})

        # Get RAG answer
        answer = await orchestrator.handle_query(question, lang)
        print(f"    Answer: {answer[:80]}...")

        results.append(
            {
                "query_id": str(i + 1),
                "query": question,
                "gt_answer": ground_truth,
                "response": answer,
                "retrieved_context": context_list
                if context_list
                else [{"doc_id": "empty", "text": "No context retrieved"}],
            }
        )

    return results


def run_rag_quality_evaluation():
    """Main entry point for RAG quality evaluation."""
    print("=" * 60)
    print("RAG QUALITY EVALUATION (RAGChecker)")
    print("=" * 60)

    # Load test data
    test_data_path = Path(__file__).parent / "test_data" / "ds_golden_v1_raw.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\nLoaded {len(test_cases)} test cases")
    print("Extractor/Checker: openai/gpt-4o-mini\n")

    # Step 1: Collect RAG responses
    print("Step 1: Collecting RAG responses...")
    raw_results = asyncio.run(collect_rag_results(test_cases))

    # Step 2: Format for RAGChecker
    rag_results = RAGResults.from_dict({"results": raw_results})

    # Step 3: Run RAGChecker evaluation
    print("\nStep 2: Running RAGChecker evaluation...")
    evaluator = RAGChecker(
        extractor_name="openai/gpt-4o-mini",
        checker_name="openai/gpt-4o-mini",
        batch_size_extractor=10,
        batch_size_checker=10,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    evaluator.evaluate(rag_results, all_metrics)

    # Step 4: Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(rag_results)

    # Extract and check thresholds
    metrics = rag_results.metrics if hasattr(rag_results, "metrics") else {}

    thresholds = {
        "overall_metrics": {"precision": 70, "recall": 60, "f1": 65},
        "generator_metrics": {"faithfulness": 65, "hallucination": 10},
    }

    all_pass = True

    if "overall_metrics" in metrics:
        om = metrics["overall_metrics"]
        for key, threshold in thresholds["overall_metrics"].items():
            val = om.get(key, 0)
            if key == "hallucination":
                status = "✅ PASS" if val <= threshold else "❌ FAIL"
                if val > threshold:
                    all_pass = False
            else:
                status = "✅ PASS" if val >= threshold else "❌ FAIL"
                if val < threshold:
                    all_pass = False
            print(f"  {key:25} {val:.1f}  (threshold: {threshold})  {status}")

    if "generator_metrics" in metrics:
        gm = metrics["generator_metrics"]
        for key, threshold in thresholds["generator_metrics"].items():
            val = gm.get(key, 0)
            if key == "hallucination":
                status = "✅ PASS" if val <= threshold else "❌ FAIL"
                if val > threshold:
                    all_pass = False
            else:
                status = "✅ PASS" if val >= threshold else "❌ FAIL"
                if val < threshold:
                    all_pass = False
            print(f"  {key:25} {val:.1f}  (threshold: {threshold})  {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL QUALITY CHECKS PASSED")
    else:
        print("❌ QUALITY CHECK FAILED - Review metrics above")
    print("=" * 60 + "\n")

    return metrics, all_pass


if __name__ == "__main__":
    run_rag_quality_evaluation()

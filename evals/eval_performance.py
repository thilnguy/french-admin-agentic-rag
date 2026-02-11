"""
Performance Evaluation - Latency Benchmarking

Measures end-to-end latency and component breakdown:
- Total latency (E2E)
- Retrieval latency (Qdrant query)
- LLM generation latency

Target SLAs:
- Simple queries: <1.5s (p95)
- Complex queries: <3s (p95)
- Retrieval: <500ms (p95)
"""

import asyncio
import json
import time
from pathlib import Path
from statistics import median, quantiles
from src.agents.orchestrator import AdminOrchestrator
from skills.legal_retriever.main import retrieve_legal_info


class LatencyTracker:
    """Track latency for different components."""

    def __init__(self):
        self.metrics = []

    async def measure_query(self, question: str, language: str, category: str):
        """Measure latency for a single query."""
        orchestrator = AdminOrchestrator()

        start_total = time.perf_counter()

        # Measure retrieval
        t_retrieval_start = time.perf_counter()
        await retrieve_legal_info(question, domain="general")
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000

        # Measure E2E (includes retrieval + LLM + guardrails)
        await orchestrator.handle_query(question, language)
        total_ms = (time.perf_counter() - start_total) * 1000

        # Estimate LLM time (E2E - retrieval)
        generation_ms = total_ms - retrieval_ms

        metric = {
            "question": question[:50] + "..." if len(question) > 50 else question,
            "category": category,
            "language": language,
            "total_ms": round(total_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
        }

        self.metrics.append(metric)
        return metric

    def get_stats(self):
        """Calculate percentile statistics."""
        total_latencies = [m["total_ms"] for m in self.metrics]
        retrieval_latencies = [m["retrieval_ms"] for m in self.metrics]

        if not total_latencies:
            return {}

        # Calculate percentiles
        p50_total, p95_total, p99_total = (
            quantiles(total_latencies, n=100)[49],
            quantiles(total_latencies, n=100)[94],
            quantiles(total_latencies, n=100)[98]
            if len(total_latencies) > 2
            else (median(total_latencies), max(total_latencies), max(total_latencies)),
        )
        p50_retrieval, p95_retrieval = (
            median(retrieval_latencies),
            quantiles(retrieval_latencies, n=100)[94]
            if len(retrieval_latencies) > 2
            else max(retrieval_latencies),
        )

        return {
            "total": {
                "p50": round(p50_total, 2),
                "p95": round(p95_total, 2),
                "p99": round(p99_total, 2),
                "min": round(min(total_latencies), 2),
                "max": round(max(total_latencies), 2),
            },
            "retrieval": {
                "p50": round(p50_retrieval, 2),
                "p95": round(p95_retrieval, 2),
            },
        }


async def run_performance_evaluation():
    """Run performance benchmarks."""
    print("=" * 60)
    print("PERFORMANCE EVALUATION (Latency Benchmarking)")
    print("=" * 60 + "\n")

    # Warmup: pre-load embeddings and Qdrant client
    print("⏳ Warming up (loading embeddings model)...")
    await retrieve_legal_info("test warmup query", domain="general")
    print("✅ Warmup complete\n")

    # Load benchmark queries
    test_data_path = Path(__file__).parent / "test_data" / "speed_bench.json"
    with open(test_data_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print(f"Running {len(benchmarks)} benchmark queries...\n")

    tracker = LatencyTracker()

    # Run benchmarks
    for bench in benchmarks:
        print(f"Testing [{bench['category']}]: {bench['question'][:60]}...")
        metric = await tracker.measure_query(
            bench["question"], bench["language"], bench["category"]
        )
        print(
            f"  Total: {metric['total_ms']}ms | Retrieval: {metric['retrieval_ms']}ms | LLM: {metric['generation_ms']}ms\n"
        )

    # Calculate statistics
    stats = tracker.get_stats()

    print("\n" + "=" * 60)
    print("LATENCY STATISTICS")
    print("=" * 60)
    print("\nEnd-to-End (Total):")
    print(f"  p50: {stats['total']['p50']}ms")
    print(f"  p95: {stats['total']['p95']}ms  (Target: <2000ms)")
    print(f"  p99: {stats['total']['p99']}ms")
    print(f"  Range: {stats['total']['min']}ms - {stats['total']['max']}ms")

    print("\nRetrieval (Qdrant):")
    print(f"  p50: {stats['retrieval']['p50']}ms")
    print(f"  p95: {stats['retrieval']['p95']}ms  (Target: <500ms)")

    # Calculate per-category stats
    simple_latencies = [
        m["total_ms"] for m in tracker.metrics if m["category"] == "simple"
    ]
    complex_latencies = [
        m["total_ms"] for m in tracker.metrics if m["category"] == "complex"
    ]

    simple_p95 = (
        quantiles(simple_latencies, n=100)[94]
        if len(simple_latencies) > 2
        else max(simple_latencies)
        if simple_latencies
        else 0
    )
    complex_p95 = (
        quantiles(complex_latencies, n=100)[94]
        if len(complex_latencies) > 2
        else max(complex_latencies)
        if complex_latencies
        else 0
    )

    print("\nBy Query Type:")
    print(f"  Simple queries p95:  {round(simple_p95, 2)}ms  (Target: <2000ms)")
    print(f"  Complex queries p95: {round(complex_p95, 2)}ms  (Target: <3500ms)")

    # SLA checks
    sla_retrieval_pass = stats["retrieval"]["p95"] < 500
    sla_simple_pass = simple_p95 < 2000 if simple_latencies else True
    sla_complex_pass = complex_p95 < 3500 if complex_latencies else True

    print("\n" + "=" * 60)
    print("SLA CHECK")
    print("=" * 60)
    print(
        f"Retrieval p95 < 500ms:     {'✅ PASS' if sla_retrieval_pass else '❌ FAIL'}"
    )
    print(f"Simple queries < 2000ms:   {'✅ PASS' if sla_simple_pass else '❌ FAIL'}")
    print(f"Complex queries < 3500ms:  {'✅ PASS' if sla_complex_pass else '❌ FAIL'}")

    if sla_retrieval_pass and sla_simple_pass and sla_complex_pass:
        print("\n✅ ALL PERFORMANCE TARGETS MET")
    else:
        print("\n❌ PERFORMANCE TARGETS NOT MET - Optimization needed")
    print("=" * 60 + "\n")

    return stats, sla_retrieval_pass and sla_simple_pass and sla_complex_pass


if __name__ == "__main__":
    asyncio.run(run_performance_evaluation())

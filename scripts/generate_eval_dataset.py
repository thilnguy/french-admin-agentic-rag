"""
Generate a blind evaluation test dataset from the topic registry.

Usage:
    python scripts/generate_blind_test_v2.py --total 100 --output evals/data/benchmarks/ds_eval_blind_v2_100.json
    python scripts/generate_blind_test_v2.py --total 50 --output evals/data/benchmarks/ds_eval_debug_50.json --model gpt-4o-mini
"""
import argparse
import asyncio
import json
import yaml
from pathlib import Path
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings


class BlindTestGenerator:
    def __init__(self, topics_path: Path, model: str, temperature: float):
        with open(topics_path, "r", encoding="utf-8") as f:
            self.registry_data = yaml.safe_load(f)["topics"]

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()

    async def generate_topic_batch(self, topic_key: str, n: int) -> List[Dict]:
        topic_info = self.registry_data[topic_key]
        prompt = ChatPromptTemplate.from_template(
            """Act as an Expert Legal Data Engineer for French Administration.
Generate {n} unique, high-quality EVALUATION test cases for the topic: "{display_name}".

TOPIC CONTEXT:
- Description: {description}
- Mandatory Variables: {mandatory_vars}

INSTRUCTIONS:
1. Scenarios: Edge-case situations NOT in standard tutorials
   (e.g., cross-border workers, refugee status, dual nationality, mid-contract disputes).
2. Language: Mix French, Vietnamese, and English within the batch.
3. Behavior mix:
   - DIRECT:  All info provided → direct, complete answer.
   - CLARIFY: Vague query → general context + ask for 2-3 mandatory variables.
   - HYBRID:  Some info given → partial answer + ask for the rest.
4. Ground truth MUST use the 3-block structure, localized to the query language:
   FR: [DONNER] / [EXPLIQUER] / [DEMANDER]
   EN: [GIVE] / [EXPLAIN] / [ASK]
   VI: [CUNG CẤP] / [GIẢI THÍCH] / [YÊU CẦU]
5. Avoid generic questions (e.g., "How do I renew my passport?").

SCHEMA — Return a JSON list of {n} objects with these exact keys:
- question: User's query (FR, VI, or EN).
- category: "{topic_key}"
- ground_truth: Ideal 3-block response in the same language as the question.
- expected_behavior: "DIRECT", "CLARIFY", or "HYBRID".
- critical_missing_info: List of variable names missing from the query (empty list if DIRECT).
- language: "fr", "vi", or "en".

Return ONLY the JSON list, no extra text or markdown."""
        )

        chain = prompt | self.llm | self.parser
        try:
            return await chain.ainvoke({
                "n": n,
                "topic_key": topic_key,
                "display_name": topic_info["display_name"],
                "description": topic_info["description"],
                "mandatory_vars": [v["name"] for v in topic_info.get("mandatory_variables", [])]
            })
        except Exception as e:
            print(f"  ⚠️  Error for topic '{topic_key}': {e}")
            return []


async def main():
    parser = argparse.ArgumentParser(
        description="Generate a blind evaluation test dataset from the topic registry."
    )
    parser.add_argument(
        "--total", type=int, default=100,
        help="Total number of test cases to generate (default: 100)."
    )
    parser.add_argument(
        "--output", type=str,
        default="evals/data/benchmarks/ds_eval_blind_registry_gpt4o.json",
        help="Output file path for the generated JSON dataset."
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model to use for generation (default: gpt-4o)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="LLM temperature for diversity (default: 0.8)."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = BlindTestGenerator(
        topics_path=Path("src/rules/topic_registry.yaml"),
        model=args.model,
        temperature=args.temperature
    )

    topics = list(generator.registry_data.keys())
    n_topics = len(topics)
    base = args.total // n_topics
    remainder = args.total % n_topics
    # Distribute remainder to first topics
    distribution = [base + (1 if i < remainder else 0) for i in range(n_topics)]

    all_cases = []
    print(f"🚀 Blind Test Generation | Total: {args.total} | Topics: {n_topics} | Model: {args.model}")
    print(f"   Output → {output_path}\n")

    for topic, n_samples in zip(topics, distribution):
        print(f"   [{topic}] Generating {n_samples} cases...", end=" ", flush=True)
        batch = await generator.generate_topic_batch(topic, n_samples)
        all_cases.extend(batch)
        print(f"✓ {len(batch)} cases (total: {len(all_cases)})")

        # Save after each topic in case of interruption
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! {len(all_cases)} cases saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

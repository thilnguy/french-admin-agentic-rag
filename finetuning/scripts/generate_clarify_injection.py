import json
import asyncio
import random
from pathlib import Path
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings

# Configuration
SEED_DATA_PATH = Path("evals/data/enriched/ds_golden_v2_enriched.json")
OUTPUT_PATH = Path("finetuning/data/clarify_injection_samples.jsonl")
MODEL_TEACHER = "gpt-4o"
TARGET_TOTAL = 100
BATCH_SIZE = 5

class ClarifyExpander:
    def __init__(self, seed_data_path: Path, model_teacher: str):
        self.llm = ChatOpenAI(
            model=model_teacher,
            temperature=0.8,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()
        
        with open(seed_data_path, "r", encoding="utf-8") as f:
            self.seed_examples = json.load(f)

    async def generate_clarify_batch(self, n: int) -> List[Dict]:
        seeds = random.sample(self.seed_examples, min(3, len(self.seed_examples)))
        
        prompt = ChatPromptTemplate.from_template(
            """Act as an AI Data Engineering Expert specializing in French Administration.
            Your task is to generate {n} unique training samples specifically designed to teach an AI when to ask CLARIFYING questions.
            
            GOAL: The user's query must be slightly vague or missing critical details (like nationality, location, or dates).
            The assistant MUST NOT give a final answer, but instead provide general context and ask for specific missing info.
            
            INSTRUCTIONS:
            - LEVEL: CLARIFY
            - Ensure the 'ground_truth' focuses on asking the right questions.
            - Provide a clear 'reasoning_outline'.
            - List the 'critical_missing_info' explicitly.
            
            SCHEMA:
            - question: The user's query (In French).
            - ground_truth: The ideal response that clarifies (In French).
            - expected_behavior: CLARIFY
            - critical_missing_info: List of items needed (In English).
            - reasoning_outline: Logic used to decide what to ask (In English).
            
            SEED EXAMPLES:
            {seeds}
            
            Return a JSON list of {n} objects."""
        )
        
        chain = prompt | self.llm | self.parser
        return await chain.ainvoke({
            "n": n,
            "seeds": json.dumps(seeds, ensure_ascii=False)
        })

async def main(seed_data_path: Path, output_path: Path, model_teacher: str, target_total: int, batch_size: int):
    expander = ClarifyExpander(seed_data_path, model_teacher)
    count = 0
    
    print(f"Generating {target_total} targeted CLARIFY samples...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        while count < target_total:
            print(f"Progress: {count}/{target_total}...")
            batch = await expander.generate_clarify_batch(batch_size)
            for sample in batch:
                if all(k in sample for k in ["question", "ground_truth", "expected_behavior"]):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
            f.flush()
            
    print(f"Success! {count} samples saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Clarify Injection Samples")
    parser.add_argument("--seed-data", type=str, default="evals/data/enriched/ds_golden_v2_enriched.json", help="Path to seed data JSON")
    parser.add_argument("--output", type=str, default="finetuning/data/clarify_injection_samples.jsonl", help="Path to output JSONL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Teacher model")
    parser.add_argument("--total", type=int, default=100, help="Total samples to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size")
    args = parser.parse_args()

    asyncio.run(main(Path(args.seed_data), Path(args.output), args.model, args.total, args.batch_size))

import json
import asyncio
import random
from pathlib import Path
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings
from src.utils.logger import logger

# Configuration
SEED_DATA_PATH = Path("evals/data/enriched/ds_golden_v2_enriched.json")
OUTPUT_PATH = Path("finetuning/data/train_augmented_500.jsonl")
MODEL_TEACHER = "gpt-4o"
TARGET_TOTAL = 500
BATCH_SIZE = 5

class AugmentationGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_TEACHER,
            temperature=0.8,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()
        
        if not SEED_DATA_PATH.exists():
            # Fallback to older name if needed, or handle error
            logger.error(f"Seed path {SEED_DATA_PATH} not found.")
            self.seed_examples = []
        else:
            with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
                self.seed_examples = json.load(f)

    async def generate_batch(self, n: int) -> List[Dict]:
        if not self.seed_examples:
            return []
            
        # Select random seeds for few-shot
        seeds = random.sample(self.seed_examples, min(5, len(self.seed_examples)))
        
        prompt = ChatPromptTemplate.from_template(
            """Act as an Expert AI Data Engineer for French Administrative Procedures.
            Using the provided seed examples, generate {n} unique, high-quality variations.
            
            INSTRUCTIONS:
            1. Scenarios: Vary the user's situation (e.g., changing nationality, age, urgency, or specific document lost).
            2. Logic: Ensure the reasoning steps (reasoning_outline) are technically correct for French law.
            3. Diversity: Mix procedures like Passport, Visa, Labor law, Social aid, and Taxes.
            4. Quality: Output must be in French for 'question' and 'ground_truth'.
            
            SCHEMA:
            - question: User's query (In French).
            - ground_truth: The direct and correct legal response (In French).
            - reason: Brief logic behind the response (In English).
            - expected_behavior: DIRECT, CLARIFY, or HYBRID.
            - critical_missing_info: List of variables needed if clarification is required.
            
            SEED EXAMPLES:
            {seeds}
            
            Return a JSON list of {n} objects."""
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            return await chain.ainvoke({
                "n": n,
                "seeds": json.dumps(seeds, ensure_ascii=False)
            })
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return []

async def main():
    generator = AugmentationGenerator()
    
    # Ensure directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Data Augmentation: {TARGET_TOTAL} samples...")
    print(f"📂 Seed Source: {SEED_DATA_PATH}")
    
    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        while count < TARGET_TOTAL:
            batch_n = min(BATCH_SIZE, TARGET_TOTAL - count)
            print(f"   Generating batch ({count}/{TARGET_TOTAL})...")
            
            batch = await generator.generate_batch(batch_n)
            
            valid_in_batch = 0
            for sample in batch:
                if all(k in sample for k in ["question", "ground_truth"]):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    valid_in_batch += 1
            
            count += valid_in_batch
            f.flush()
            
    print(f"✅ Augmentation complete! Stored in: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())

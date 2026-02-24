import json
import asyncio
import random
import os
from pathlib import Path
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings
from src.utils.logger import logger

# Configuration
SEED_DATA_PATH = Path("evals/data/enriched/ds_golden_v1_enriched.json")
OUTPUT_PATH = Path("finetuning/data/self_instruct_samples.jsonl")
MODEL_TEACHER = "gpt-4o"
TARGET_TOTAL = 300
BATCH_SIZE = 5

COMPLEXITY_DIST = {
    "BASIC": 0.30,      # Expected 90
    "CLARIFY": 0.40,    # Expected 120
    "COMPLEX": 0.30     # Expected 90
}

TOPICS = [
    "Titre de séjour (Etudiant, Salarié, Famille)",
    "Naturalisation française",
    "Permis de conduire (Echange, Perte, Examen)",
    "Passeport và Carte d'identité",
    "Regroupement familial"
]

SCENARIOS = [
    "Mất giấy tờ (Perte/Vol)",
    "Hết hạn (Expiration/Renouvellement)",
    "Thiếu hồ sơ (Dossier incomplet)",
    "Quốc tịch EU/Non-EU",
    "Tình huống khẩn cấp (Urgence)"
]

class SelfInstructExpander:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_TEACHER,
            temperature=0.7, # Higher temperature for diversity
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()
        
        with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
            self.seed_examples = json.load(f)

    async def generate_batch(self, complexity: str, n: int) -> List[Dict]:
        topic = random.choice(TOPICS)
        scenario = random.choice(SCENARIOS)
        
        # Select random seeds for few-shot
        seeds = random.sample(self.seed_examples, min(3, len(self.seed_examples)))
        
        prompt = ChatPromptTemplate.from_template(
            """Act as an AI Data Engineering Expert specializing in French Administration.
            Your task is to generate {n} unique, high-quality training samples for a 'Student' model.
            
            LEVEL: {complexity}
            PRIMARY TOPIC: {topic}
            SCENARIO: {scenario}
            
            INSTRUCTIONS:
            Phase 1: Diversity Generation: Create unique situations (lost docs, expired, missing info, EU vs Non-EU).
            Phase 2: Reasoning Consistency: Ensure the answer matches the question perfectly.
            Phase 3: Complexity Scaling:
            - BASIC: Direct answer, all info provided.
            - CLARIFY: User query is vague/missing info, answer must ask for specific details.
            - COMPLEX: Edge cases, legal exceptions, or urgent situations.
            
            SCHEMA:
            - question: The user's query (In French).
            - ground_truth: Precise legal answer (In French).
            - context_keywords: 3-5 keywords (In English).
            - expected_behavior: CLARIFY, ANSWER, or HYBRID.
            - critical_missing_info: List of missing info (In English).
            - reasoning_outline: Condensed logic steps (List in English).
            
            SEED EXAMPLES:
            {seeds}
            
            Return a JSON list of {n} objects strictly following the schema."""
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            return await chain.ainvoke({
                "complexity": complexity,
                "topic": topic,
                "scenario": scenario,
                "n": n,
                "seeds": json.dumps(seeds, ensure_ascii=False)
            })
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return []

async def main():
    expander = SelfInstructExpander()
    
    counts = {"BASIC": 0, "CLARIFY": 0, "COMPLEX": 0}
    targets = {k: int(TARGET_TOTAL * v) for k, v in COMPLEXITY_DIST.items()}
    
    print(f"Starting Self-Instruct Expansion to {TARGET_TOTAL} samples...")
    print(f"Targets: {targets}")
    
    # Ensure directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Mode 'w' to reset if needed, or 'a' to resume
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        while sum(counts.values()) < TARGET_TOTAL:
            # Determine which complexity to target next
            remaining = {k: targets[k] - counts[k] for k in counts if counts[k] < targets[k]}
            if not remaining: break
            
            complexity = random.choice(list(remaining.keys()))
            n = min(BATCH_SIZE, remaining[complexity])
            
            print(f"Generating batch of {n} [{complexity}] samples... (Total: {sum(counts.values())}/{TARGET_TOTAL})")
            
            batch = await expander.generate_batch(complexity, n)
            
            valid_samples = 0
            for sample in batch:
                # Basic validation
                if all(k in sample for k in ["question", "ground_truth", "reasoning_outline"]):
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    valid_samples += 1
            
            counts[complexity] += valid_samples
            f.flush()
            
    print(f"Expansion complete! Final counts: {counts}")
    print(f"Stored in: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())

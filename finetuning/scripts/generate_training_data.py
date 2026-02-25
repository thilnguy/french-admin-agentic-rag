import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings
from src.utils.logger import logger

class DataGenerator:
    def __init__(self, model_teacher: str, temperature: float, seed_data_path: Optional[Path] = None):
        self.llm = ChatOpenAI(
            model=model_teacher,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()
        
        self.seed_examples = []
        if seed_data_path and seed_data_path.exists():
            with open(seed_data_path, "r", encoding="utf-8") as f:
                self.seed_examples = json.load(f)
        elif seed_data_path:
            logger.warning(f"Seed data path {seed_data_path} not found. Some strategies may fail or output empty results.")

    async def _invoke_chain(self, prompt: ChatPromptTemplate, params: dict, retries: int = 3) -> List[Dict]:
        chain = prompt | self.llm | self.parser
        for attempt in range(retries):
            try:
                return await chain.ainvoke(params)
            except Exception as e:
                logger.error(f"Batch generation error (Attempt {attempt+1}/{retries}): {e}")
                await asyncio.sleep(2 ** attempt)
        return []

    # --- Strategy Methods ---

    async def generate_augment_batch(self, n: int) -> List[Dict]:
        seeds = random.sample(self.seed_examples, min(5, len(self.seed_examples))) if self.seed_examples else []
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
        return await self._invoke_chain(prompt, {"n": n, "seeds": json.dumps(seeds, ensure_ascii=False)})

    async def generate_clarify_batch(self, n: int) -> List[Dict]:
        seeds = random.sample(self.seed_examples, min(3, len(self.seed_examples))) if self.seed_examples else []
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
        return await self._invoke_chain(prompt, {"n": n, "seeds": json.dumps(seeds, ensure_ascii=False)})

    async def generate_focus_batch(self, topic: str, n: int) -> List[Dict]:
        prompt = ChatPromptTemplate.from_template(
            """Act as an Expert Administrative Assistant for French Public Services.
            Your goal is to generate {n} extremely precise training samples for a model that has been incorrectly refusing these topics.
            
            TOPIC: {topic}
            
            Focus specifically on:
            - EXACT COSTS (e.g., 225€ for residency, 86€ for passport).
            - DIRECT ANSWERS (no refusal, no 'I can't help').
            - FORMAT: CoT reasoning followed by the direct answer.
            
            SCHEMA:
            - question: The user's query in French.
            - ground_truth: The direct, precise answer in French.
            - reasoning_outline: Condensed logic steps in English (e.g., ["Identity check", "Cost lookup", "Form 123 usage"]).
            
            Return a JSON list of {n} samples."""
        )
        return await self._invoke_chain(prompt, {"topic": topic, "n": n})

    async def generate_self_instruct_batch(self, complexity: str, n: int) -> List[Dict]:
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
        topic = random.choice(TOPICS)
        scenario = random.choice(SCENARIOS)
        seeds = random.sample(self.seed_examples, min(3, len(self.seed_examples))) if self.seed_examples else []
        
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
        return await self._invoke_chain(prompt, {"complexity": complexity, "topic": topic, "scenario": scenario, "n": n, "seeds": json.dumps(seeds, ensure_ascii=False)})

async def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data for fine-tuning")
    parser.add_argument("--strategy", type=str, required=True, choices=["augment", "clarify", "focus", "self-instruct"], help="The generation strategy to use")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--seed-data", type=str, default="evals/data/enriched/ds_golden_v2_enriched.json", help="Path to seed data JSON (if required by strategy)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Teacher model")
    parser.add_argument("--total", type=int, default=100, help="Total samples to generate")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size per LLM call")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for LLM generation")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator = DataGenerator(args.model, args.temperature, Path(args.seed_data))
    count = 0
    
    print(f"🚀 Starting Data Generation | Strategy: {args.strategy} | Target: {args.total} samples")
    
    with open(output_path, "w", encoding="utf-8") as f:
        if args.strategy == "self-instruct":
            COMPLEXITY_DIST = {"BASIC": 0.30, "CLARIFY": 0.40, "COMPLEX": 0.30}
            counts = {k: 0 for k in COMPLEXITY_DIST}
            targets = {k: int(args.total * v) for k, v in COMPLEXITY_DIST.items()}
            
            while sum(counts.values()) < args.total:
                remaining = {k: targets[k] - counts[k] for k in counts if counts[k] < targets[k]}
                if not remaining: break
                
                complexity = random.choice(list(remaining.keys()))
                n = min(args.batch_size, remaining[complexity])
                print(f"   Generating batch ({n}) [{complexity}]... (Total: {sum(counts.values())}/{args.total})")
                
                batch = await generator.generate_self_instruct_batch(complexity, n)
                for sample in batch:
                    if "question" in sample and "ground_truth" in sample:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        counts[complexity] += 1
                f.flush()
            print(f"✅ Self-Instruct complete! Final counts: {counts}")
            count = sum(counts.values())
            
        elif args.strategy == "focus":
            FOCUS_TOPICS = [
                "Permis de conduire (Echange, Perte, Coût)",
                "Coût d'un titre de séjour (Timbres fiscaux)",
                "Passeport (Tarifs, Renouvellement, Timbre fiscal)",
                "Certificat d'immatriculation (Carte grise, Coût)"
            ]
            samples_per_topic = max(1, args.total // len(FOCUS_TOPICS))
            
            for topic in FOCUS_TOPICS:
                print(f"   Targeting topic: {topic}")
                # We may need multiple batches per topic if samples_per_topic > batch_size
                generated_for_topic = 0
                while generated_for_topic < samples_per_topic and count < args.total:
                    n = min(args.batch_size, samples_per_topic - generated_for_topic, args.total - count)
                    batch = await generator.generate_focus_batch(topic, n)
                    for sample in batch:
                        if "question" in sample and "ground_truth" in sample:
                            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            count += 1
                            generated_for_topic += 1
                    f.flush()
                
        else:
            # augment or clarify
            while count < args.total:
                batch_n = min(args.batch_size, args.total - count)
                print(f"   Generating batch ({count}/{args.total})...")
                
                if args.strategy == "augment":
                    batch = await generator.generate_augment_batch(batch_n)
                else: # clarify
                    batch = await generator.generate_clarify_batch(batch_n)
                
                for sample in batch:
                    if "question" in sample and "ground_truth" in sample:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        count += 1
                f.flush()
                
    print(f"✅ Success! {count} samples saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())

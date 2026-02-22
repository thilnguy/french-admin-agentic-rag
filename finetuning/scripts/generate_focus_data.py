import json
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import settings
from src.utils.logger import logger

# TARGET TOPICS previously causing refusals
FOCUS_TOPICS = [
    "Permis de conduire (Echange, Perte, Coût)",
    "Coût d'un titre de séjour (Timbres fiscaux)",
    "Passeport (Tarifs, Renouvellement, Timbre fiscal)",
    "Certificat d'immatriculation (Carte grise, Coût)"
]

OUTPUT_PATH = Path("finetuning/data/focus_samples.jsonl")
MODEL_TEACHER = "gpt-4o"
TARGET_SAMPLES = 50

class FocusDataGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_TEACHER,
            temperature=0.8,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = JsonOutputParser()

    async def generate_focus_batch(self, topic: str, n: int):
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
        
        chain = prompt | self.llm | self.parser
        try:
            return await chain.ainvoke({"topic": topic, "n": n})
        except Exception as e:
            logger.error(f"Focus generation failed for {topic}: {e}")
            return []

async def main():
    generator = FocusDataGenerator()
    samples_per_topic = TARGET_SAMPLES // len(FOCUS_TOPICS)
    
    print(f"Generating {TARGET_SAMPLES} focus samples for refusal correction...")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for topic in FOCUS_TOPICS:
            print(f"Targeting topic: {topic}")
            batch = await generator.generate_focus_batch(topic, samples_per_topic)
            for sample in batch:
                if "question" in sample and "ground_truth" in sample:
                    # Format for MLX (using human/assistant structure or the custom expert format)
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"Generated {len(batch)} samples for {topic}")

    print(f"Focus data stored in: {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())

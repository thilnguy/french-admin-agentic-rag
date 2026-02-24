import asyncio
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import settings


# Define the Output Schema
class EnrichedTestCase(BaseModel):
    expected_behavior: Literal["DIRECT", "CLARIFY", "HYBRID"] = Field(
        ..., description="The ideal behavior for the agent."
    )
    critical_missing_info: List[str] = Field(
        default_factory=list,
        description="List of specific details missing from the question (e.g., 'nationality', 'age').",
    )
    reasoning: str = Field(..., description="Brief reason for this classification.")


async def enrich_data():
    input_path = Path(__file__).parent.parent / "data" / "raw" / "ds_golden_v2_raw.json"
    output_path = Path(__file__).parent.parent / "data" / "enriched" / "ds_golden_v2_enriched.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)

    parser = PydanticOutputParser(pydantic_object=EnrichedTestCase)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert Data Labeler for a French Administration RAG Agent.
your task is to analyze a User Question and the Ground Truth Answer to determine how an ideal AI Agent should respond.

### Classification Rules:
1. **DIRECT**: The question is specific, complete, and factual (e.g., "Cost of passport", "Address of Mairie"). No personal details needed.
2. **CLARIFY**: The question is too vague to answer without specific details (e.g., "I want a visa" -> Needs nationality, duration, purpose).
3. **HYBRID**: The question asks about a procedure that has a general answer BUT depends on details for the specific steps (e.g., "How to renew passport?" -> Give general rules + ask for nationality/expiration). **Prefer HYBRID for most procedure questions.**

### Output Format:
- `expected_behavior`: DIRECT, CLARIFY, or HYBRID.
- `critical_missing_info`: List of specific variables needed to give a *perfect* answer (e.g., ["nationality", "residence_status"]). Leave empty if DIRECT.
- `reasoning`: Short explanation.

{format_instructions}
""",
            ),
            (
                "user",
                """
Question: {question}
Ground Truth: {ground_truth}
Language: {language}
""",
            ),
        ]
    )

    chain = prompt | llm | parser

    enriched_data = []
    print(f"Enriching {len(data)} test cases...")

    for i, item in enumerate(data):
        print(f"Processing [{i+1}/{len(data)}]: {item['question'][:50]}...")
        try:
            result = await chain.ainvoke(
                {
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "language": item.get("language", "fr"),
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            # Merge result into item
            new_item = item.copy()
            new_item["expected_behavior"] = result.expected_behavior
            new_item["critical_missing_info"] = result.critical_missing_info

            enriched_data.append(new_item)
        except Exception as e:
            print(f"Error classifying item {i}: {e}")
            enriched_data.append(item)  # Keep original if fail

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Created {output_path} with {len(enriched_data)} enriched cases.")


if __name__ == "__main__":
    asyncio.run(enrich_data())

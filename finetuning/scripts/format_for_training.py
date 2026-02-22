import json
from pathlib import Path

# Configuration
INPUT_PATH = Path("finetuning/data/final_train_raw.jsonl")
OUTPUT_PATH = Path("finetuning/data/train_expert_formatted.jsonl")

def format_sample(sample: dict) -> dict:
    """
    Converts a self-instruct sample into OpenAI ChatML format 
    with reasoning prepended to the ground truth.
    """
    # Build thinking trace
    reasoning = "\n".join([f"- {step}" for step in sample.get("reasoning_outline", [])])
    
    # Format assistant response as <thinking> reasoning </thinking> answer
    assistant_content = f"<thinking>\n{reasoning}\n</thinking>\n\n{sample['ground_truth']}"
    
    # Optional: Add missing info if expected behavior is CLARIFY
    if sample.get("expected_behavior") == "CLARIFY" and sample.get("critical_missing_info"):
        missing = ", ".join(sample["critical_missing_info"])
        assistant_content += f"\n\n(Information missing: {missing})"

    return {
        "messages": [
            {"role": "system", "content": "You are a French Administration Assistant. Reason step-by-step before answering."},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    print(f"Formatting expert samples from {INPUT_PATH}...")
    
    formatted_count = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            try:
                sample = json.loads(line)
                formatted = format_sample(sample)
                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                formatted_count += 1
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    print(f"Successfully formatted {formatted_count} samples for training.")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

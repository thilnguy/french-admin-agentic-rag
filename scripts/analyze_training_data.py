import json
from collections import Counter
from pathlib import Path

def analyze_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        return f"File {file_path} not found."
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    total = len(data)
    if total == 0:
        return f"File {file_path} is empty."

    # Check if it's ChatML or Raw
    is_chatml = 'messages' in data[0]
    
    q_lengths = []
    has_reasoning = 0
    categories = Counter()
    behaviors = Counter()
    languages = Counter()

    if is_chatml:
        for item in data:
            for msg in item['messages']:
                if msg['role'] == 'user':
                    q_lengths.append(len(msg['content']))
                if msg['role'] == 'assistant' and '<thinking>' in msg['content']:
                    has_reasoning += 1
    else:
        for item in data:
            categories[item.get('category', 'Unknown')] += 1
            behaviors[item.get('expected_behavior', 'Unknown')] += 1
            languages[item.get('language', 'Unknown')] += 1
            q_lengths.append(len(item.get('question', '')))
            if 'reasoning_outline' in item or '<thinking>' in str(item):
                has_reasoning += 1
    
    avg_q_len = sum(q_lengths) / total if total > 0 else 0
    
    report = f"Analysis for: {file_path}\n"
    report += f"Total Samples: {total}\n"
    report += f"Reasoning/CoT presence: {has_reasoning}/{total}\n"
    report += f"Avg Question Length: {avg_q_len:.1f} chars\n\n"
    
    if not is_chatml:
        report += "Categories Distribution:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            report += f"  - {cat}: {count} ({count/total*100:.1f}%)\n"
            
        report += "\nBehaviors Distribution:\n"
        for beh, count in behaviors.items():
            report += f"  - {beh}: {count} ({count/total*100:.1f}%)\n"
            
        report += "\nLanguages Distribution:\n"
        for lang, count in languages.items():
            report += f"  - {lang}: {count} ({count/total*100:.1f}%)\n"
        
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training data jsonl files")
    parser.add_argument("--inputs", nargs='+', required=True, help="Path(s) to the jsonl files to analyze")
    args = parser.parse_args()

    for i, file_path in enumerate(args.inputs):
        if i > 0:
            print("\n" + "="*50 + "\n")
        print(analyze_dataset(file_path))


import json
from collections import defaultdict

def analyze_results():
    with open('evals/results/llm_judge_results_final_100_registry.json', 'r') as f:
        data = json.load(f)
    
    total = len(data)
    total_score = sum(v['verdict']['score'] for v in data)
    avg_score = total_score / total
    
    stats = defaultdict(list)
    failures = []
    
    for v in data:
        stats[v['case']['category']].append(v['verdict']['score'])
        if v['verdict']['score'] < 10:
            failures.append({
                'q': v['case']['question'],
                'cat': v['case']['category'],
                'score': v['verdict']['score'],
                'reason': v['verdict']['reasoning']
            })
            
    print(f"Total Cases: {total}")
    print(f"Overall Average: {avg_score:.2f}/10")
    print("\nTopic Breakdown:")
    for cat, scores in stats.items():
        avg_cat = sum(scores) / len(scores)
        print(f" - {cat}: {avg_cat:.2f}/10 ({len(scores)} cases)")
        
    print("\n--- TOP FAILURES ---")
    for f in failures[:15]:
        print(f"[{f['score']}/10] Topic: {f['cat']}")
        print(f"Q: {f['q']}")
        print(f"Reason: {f['reason']}\n")

if __name__ == '__main__':
    analyze_results()

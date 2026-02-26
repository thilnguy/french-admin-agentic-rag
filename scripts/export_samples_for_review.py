import os
import json
import csv
from datetime import datetime
import argparse

def export_audit_logs(input_log: str, output_csv: str, limit: int = 50):
    """
    Reads the most recent lines from the audit log and exports them
    to a CSV format suitable for human expert review.
    """
    if not os.path.exists(input_log):
        print(f"Error: Audit log not found at {input_log}")
        return

    # Read logs (last 'limit' entries)
    logs = []
    with open(input_log, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "audit_data" in data:
                    logs.append(data)
            except json.JSONDecodeError:
                continue

    # Take the most recent 'limit' logs
    logs = logs[-limit:]
    
    if not logs:
        print("No valid audit logs found to export.")
        return

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Timestamp', 
            'Session ID', 
            'Query', 
            'Detected Intent', 
            'Detected Language', 
            'Response Length',
            'Expert Rating (1-5)',
            'Expert Intent Correction',
            'Expert Comments'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            audit = log["audit_data"]
            writer.writerow({
                'Timestamp': log.get('timestamp', ''),
                'Session ID': audit.get('session_id', ''),
                'Query': audit.get('query', ''),
                'Detected Intent': audit.get('intent', ''),
                'Detected Language': audit.get('language', ''),
                'Response Length': audit.get('response_length', 0),
                'Expert Rating (1-5)': '',
                'Expert Intent Correction': '',
                'Expert Comments': ''
            })

    print(f"Successfully exported {len(logs)} samples to {output_csv} for expert review.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export recent audit logs for expert review.")
    parser.add_argument("--limit", type=int, default=50, help="Number of recent logs to export.")
    parser.add_argument("--output", type=str, default="evals/data/expert_review_samples.csv", help="Output CSV path.")
    args = parser.parse_args()
    
    export_audit_logs("logs/audit.log", args.output, args.limit)

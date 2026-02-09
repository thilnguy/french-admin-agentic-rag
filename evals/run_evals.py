import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from src.agents.orchestrator import AdminOrchestrator
from dotenv import load_dotenv

load_dotenv()

def run_evaluation():
    orchestrator = AdminOrchestrator()
    
    # Simple test set
    test_questions = [
        "Comment renouveler mon passeport ?",
        "Quelles sont các bước để xin thẻ cư trú (titre de séjour) ?",
        "How to get a French driver's license?"
    ]
    
    # Collecting results
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [
            "Il faut s'adresser à une mairie équipée d'une station d'enregistrement.",
            "Il faut déposer un dossier à la préfecture ou sous-préfecture de son domicile.",
            "Il faut justifier d'une résidence normale en France và vượt qua kỳ thi sát hạch nếu cần."
        ]
    }
    
    for q in test_questions:
        # Mocking the orchestrator run for evaluation purposes
        # In a real scenario, we'd capture the intermediate context
        res = orchestrator.handle_query(q)
        data_samples["question"].append(q)
        data_samples["answer"].append(res)
        data_samples["contexts"].append(["Retrieved context placeholder"]) # Capture from actual retriever in production

    dataset = Dataset.from_dict(data_samples)
    
    # RAGAS metrics
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print("Evaluation Results:", score)

if __name__ == "__main__":
    run_evaluation()

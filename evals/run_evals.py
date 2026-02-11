import asyncio
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from src.agents.orchestrator import AdminOrchestrator


async def collect_responses(test_questions):
    orchestrator = AdminOrchestrator()
    answers = []
    contexts = []

    for q in test_questions:
        # In a real scenario, we'd capture the intermediate context
        # For now, running end-to-end
        res = await orchestrator.handle_query(q)
        answers.append(res)
        # Ragas expects contexts as list of strings
        contexts.append(["Retrieved context placeholder"])

    return answers, contexts


def run_evaluation():
    # Simple test set
    test_questions = [
        "Comment renouveler mon passeport ?",
        "Quelles sont các bước để xin thẻ cư trú (titre de séjour) ?",
        "How to get a French driver's license?",
    ]

    ground_truth = [
        "Il faut s'adresser à une mairie équipée d'une station d'enregistrement.",
        "Il faut déposer un dossier à la préfecture ou sous-préfecture de son domicile.",
        "Il faut justifier d'une résidence normale en France và vượt qua kỳ thi sát hạch nếu cần.",
    ]

    print("Collecting responses (Async)...")
    answers, contexts = asyncio.run(collect_responses(test_questions))

    # Collecting results
    data_samples = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }

    dataset = Dataset.from_dict(data_samples)

    # RAGAS metrics
    print("Running Ragas evaluation...")
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print("Evaluation Results:", score)


if __name__ == "__main__":
    run_evaluation()

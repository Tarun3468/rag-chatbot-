"""
Evaluate retrieval + generation quality using RAGAS.

Fill in eval_data with 10-15 real question/ground-truth pairs from your
own documents, then run this before and after adding hybrid search
(app/retrieval/hybrid.py) to get before/after numbers for your resume.

Usage: python evaluate.py
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.retrieval.chain import answer_question

# Replace with your own questions and known-correct answers.
eval_questions = [
    "What is the main topic of the document?",
    # add 10-15 real questions here
]
eval_ground_truths = [
    "Replace with the correct answer for the question above",
    # matching ground truths here
]

if __name__ == "__main__":
    answers = []
    contexts = []

    for q in eval_questions:
        result = answer_question(q)
        answers.append(result["answer"])
        # sources here are metadata dicts; for RAGAS we need the actual
        # retrieved text - adjust answer_question() to also return
        # page_content if you want full context evaluation.
        contexts.append([str(s) for s in result["sources"]])

    eval_data = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": eval_ground_truths,
    }

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    print(result)

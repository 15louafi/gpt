"""Evaluation framework — measures RAG system quality on sample inputs.

Usage: uv run python -m src.evaluate
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .generation import StructuredGenerator
from .pipeline import Pipeline
from .reranking import LLMReranker
from .retrieval import ExpandedRetriever
from .retrieval.semantic import SemanticRetriever

VALIDATION_SET = [
    ("Dyspnée à l'effort", ["R06.0"]),
    ("Toux purulente", ["R05"]),
    ("Fièvre", ["R50"]),
    ("Œdème des membres inférieurs", ["R60"]),
    ("Hyponatrémie", ["E87.1"]),
    ("Hypercalcémie", ["E83.5"]),
    ("Tachycardie", ["R00.0"]),
    ("Détresse respiratoire aiguë", ["J96.0", "J80"]),
    ("Hypertension artérielle", ["I10"]),
    ("Diabète de type 2", ["E11"]),
    ("Fibrillation auriculaire", ["I48"]),
    ("Infection pulmonaire à Haemophilus influenzae", ["J14"]),
    ("Dyslipidémie", ["E78"]),
    ("Insuffisance rénale aiguë", ["N17"]),
    ("Pneumopathie à Haemophilus influenzae", ["J14"]),
]


def _code_matches(suggested: str, expected: str) -> bool:
    return suggested.startswith(expected) or expected.startswith(suggested)


def run_evaluation(pipeline: Pipeline) -> dict:
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = []
    for i, (text, expected) in enumerate(VALIDATION_SET):
        print(f"\n[{i + 1}/{len(VALIDATION_SET)}] {text}")
        start = time.time()
        response = pipeline.query(text)
        latency = time.time() - start

        suggested = [c.code for c in response.codes]
        hit3 = any(any(_code_matches(s, e) for e in expected) for s in suggested[:3])
        hit5 = any(any(_code_matches(s, e) for e in expected) for s in suggested[:5])
        recall = sum(1 for e in expected if any(_code_matches(s, e) for s in suggested)) / len(
            expected
        )

        print(
            f"  {'✅' if hit3 else '❌'} Expected: {expected} | Got: {suggested[:3]} | {latency:.1f}s"  # noqa: E501
        )
        results.append(
            {
                "input": text,
                "expected": expected,
                "suggested": suggested[:5],
                "hit_at_3": hit3,
                "hit_at_5": hit5,
                "recall": recall,
                "latency": round(latency, 2),
            }
        )

    n = len(results)
    metrics = {
        "hit_at_3": sum(r["hit_at_3"] for r in results) / n,
        "hit_at_5": sum(r["hit_at_5"] for r in results) / n,
        "mean_recall": sum(r["recall"] for r in results) / n,
        "mean_latency": sum(r["latency"] for r in results) / n,
        "total_cases": n,
    }

    print(f"\n{'=' * 60}")
    print("RESULTS")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2%}" if isinstance(v, float) else f"  {k}: {v}")

    with open("data/evaluation_results.json", "w") as f:
        json.dump({"metrics": metrics, "details": results}, f, ensure_ascii=False, indent=2)

    return metrics


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY required")
        return

    client = OpenAI(api_key=api_key)
    semantic = SemanticRetriever(client)
    if semantic.count() == 0:
        print("ERROR: Vector store empty. Run ingestion first.")
        return

    pipeline = Pipeline(
        retriever=ExpandedRetriever(semantic, client),
        reranker=LLMReranker(client),
        generator=StructuredGenerator(client),
    )
    run_evaluation(pipeline)


if __name__ == "__main__":
    main()

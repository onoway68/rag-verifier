import pytest

from embedding_provider import (
    FakeEmbeddingProvider
)
from retrieval_evaluator import (
    RetrievalEvaluator
)
from retriever import Retriever


def test_evaluator_measures_actual_retriever_rankings():
    chunks = {
        "mi-001": (
            "Heart attack involves blocked "
            "blood flow."
        ),
        "pna-001": (
            "Pneumonia affects the lungs."
        ),
        "htn-001": (
            "Hypertension means elevated "
            "blood pressure."
        )
    }

    provider = FakeEmbeddingProvider(
        embeddings={
            "heart attack question": [
                1.0,
                0.0
            ],
            "ambiguous pneumonia question": [
                0.8,
                0.6
            ],
            chunks["mi-001"]: [
                1.0,
                0.0
            ],
            chunks["pna-001"]: [
                0.0,
                1.0
            ],
            chunks["htn-001"]: [
                -1.0,
                0.0
            ]
        }
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=provider
    )

    evaluator = RetrievalEvaluator()

    cases = [
        {
            "query": "heart attack question",
            "relevant_ids": {
                "mi-001"
            }
        },
        {
            "query": (
                "ambiguous pneumonia question"
            ),
            "relevant_ids": {
                "pna-001"
            }
        }
    ]

    result = evaluator.evaluate_retriever(
        retriever=retriever,
        cases=cases,
        k=2
    )

    assert result["queries"][0][
        "retrieved_ids"
    ][0] == "mi-001"

    assert result["queries"][1][
        "retrieved_ids"
    ] == [
        "mi-001",
        "pna-001"
    ]

    assert result["summary"][
        "mean_precision_at_k"
    ] == pytest.approx(
        0.5
    )

    assert result["summary"][
        "mean_recall_at_k"
    ] == pytest.approx(
        1.0
    )

    assert result["summary"]["mrr"] == pytest.approx(
        0.75
    )

    assert result["summary"][
        "query_count"
    ] == 2

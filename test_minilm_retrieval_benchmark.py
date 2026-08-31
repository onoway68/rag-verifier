import pytest

from embedding_provider import (
    SentenceTransformerEmbeddingProvider
)
from retrieval_evaluator import (
    RetrievalEvaluator
)
from retriever import Retriever


@pytest.mark.integration
def test_minilm_retrieval_benchmark():
    chunks = {
        "mi-001": (
            "Myocardial infarction occurs when "
            "blood flow to part of the heart "
            "muscle is blocked or severely reduced."
        ),
        "mi-002": (
            "Common symptoms of myocardial "
            "infarction include chest pressure, "
            "shortness of breath, nausea, and "
            "sweating."
        ),
        "pna-001": (
            "Pneumonia is an infection that "
            "inflames the air sacs in one or "
            "both lungs."
        ),
        "pna-002": (
            "Symptoms of pneumonia can include "
            "cough, fever, chills, and difficulty "
            "breathing."
        ),
        "htn-001": (
            "Hypertension is persistent elevation "
            "of blood pressure in the arteries."
        ),
        "htn-002": (
            "Long-term hypertension can increase "
            "the risk of heart disease, stroke, "
            "and kidney disease."
        )
    }

    cases = [
        {
            "query": (
                "What happens when blood flow "
                "to the heart muscle becomes "
                "blocked?"
            ),
            "relevant_ids": {
                "mi-001"
            }
        },
        {
            "query": (
                "What symptoms can occur during "
                "a heart attack?"
            ),
            "relevant_ids": {
                "mi-002"
            }
        },
        {
            "query": (
                "What kind of infection inflames "
                "the air sacs of the lungs?"
            ),
            "relevant_ids": {
                "pna-001"
            }
        },
        {
            "query": (
                "What symptoms are commonly seen "
                "with pneumonia?"
            ),
            "relevant_ids": {
                "pna-002"
            }
        },
        {
            "query": (
                "What condition means persistently "
                "elevated arterial blood pressure?"
            ),
            "relevant_ids": {
                "htn-001"
            }
        },
        {
            "query": (
                "What complications can develop "
                "from long-term high blood pressure?"
            ),
            "relevant_ids": {
                "htn-002"
            }
        }
    ]

    provider = SentenceTransformerEmbeddingProvider(
        model_id=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=provider
    )

    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate_retriever(
        retriever=retriever,
        cases=cases,
        k=3
    )

    print(
        "\nMiniLM retrieval benchmark summary:",
        result["summary"]
    )

    for query_result in result["queries"]:
        print(
            "\nQuery:",
            query_result["query"]
        )
        print(
            "Retrieved:",
            query_result["retrieved_ids"]
        )
        print(
            "Metrics:",
            query_result["metrics"]
        )

    assert result["summary"][
        "query_count"
    ] == 6

    assert result["summary"][
        "mean_recall_at_k"
    ] >= 0.80

    assert result["summary"]["mrr"] >= 0.80

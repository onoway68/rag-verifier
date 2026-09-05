import pytest

from healthcare_retrieval_benchmark import (
    HealthcareRetrievalBenchmark
)
from model_qualification import ModelQualificationRecordBuilder


class StaticRetriever:
    def retrieve(self, query, top_k):
        return [{"chunk_id": "DOC-1"}]


def test_benchmark_result_copies_relevant_ids():
    relevant_ids = ["DOC-1"]
    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": relevant_ids
    }

    benchmark = HealthcareRetrievalBenchmark(
        retriever=StaticRetriever(),
        top_k=1
    )

    result = benchmark.evaluate([query])
    relevant_ids[0] = "TAMPERED"

    assert result["queries"][0]["relevant_ids"] == ["DOC-1"]
    assert result["queries"][0]["recall_at_k"] == pytest.approx(1.0)


def test_unhashable_qualification_status_is_rejected_as_value_error():
    class InvalidStatusPolicy:
        def decide(self, metrics):
            return {
                "status": ["PASS"],
                "reason": "invalid"
            }

    builder = ModelQualificationRecordBuilder(
        policy=InvalidStatusPolicy()
    )

    with pytest.raises(
        ValueError,
        match="qualification output is invalid"
    ):
        builder.build(
            model={
                "model_id": "example/model",
                "model_revision": "abc123",
                "provider_type": "example",
                "embedding_dimension": 384
            },
            benchmark={
                "benchmark_id": "healthcare",
                "benchmark_version": "1.0",
                "top_k": 1
            },
            metrics={
                "query_count": 1,
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "precision_at_k": 1.0
            }
        )

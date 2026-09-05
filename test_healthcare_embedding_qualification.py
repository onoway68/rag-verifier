import pytest

from embedding_provider import FakeEmbeddingProvider
from healthcare_embedding_qualification import (
    HealthcareEmbeddingQualificationRunner,
)
from model_qualification import ModelQualificationPolicy


CHUNKS = {
    "DOC-HTN": "Hypertension is persistent elevation of arterial blood pressure.",
    "DOC-DM": "Diabetes mellitus is characterized by chronic hyperglycemia.",
    "DOC-ASTHMA": "Asthma is a chronic inflammatory airway disease.",
}

QUERIES = [
    {
        "query_id": "Q-HTN",
        "text": "What condition causes persistently elevated blood pressure?",
        "relevant_ids": ["DOC-HTN"],
    },
    {
        "query_id": "Q-DM",
        "text": "Which condition is defined by chronic hyperglycemia?",
        "relevant_ids": ["DOC-DM"],
    },
]

MODEL = {
    "model_id": "example/healthcare-embedding-model",
    "model_revision": "revision-001",
    "provider_type": "sentence-transformer",
    "embedding_dimension": 3,
}


def _policy():
    return ModelQualificationPolicy(
        pass_thresholds={
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "precision_at_k": 0.5,
        },
        review_thresholds={
            "recall_at_k": 0.5,
            "mrr": 0.5,
            "precision_at_k": 0.25,
        },
    )


def _perfect_provider():
    provider = FakeEmbeddingProvider()

    provider.set_embedding(CHUNKS["DOC-HTN"], [1.0, 0.0, 0.0])
    provider.set_embedding(CHUNKS["DOC-DM"], [0.0, 1.0, 0.0])
    provider.set_embedding(CHUNKS["DOC-ASTHMA"], [0.0, 0.0, 1.0])
    provider.set_embedding(QUERIES[0]["text"], [1.0, 0.0, 0.0])
    provider.set_embedding(QUERIES[1]["text"], [0.0, 1.0, 0.0])

    return provider


def test_runner_builds_governed_pass_record():
    captured = []

    def provider_factory(model):
        captured.append(model)
        return _perfect_provider()

    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=provider_factory,
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    record = runner.qualify(MODEL, CHUNKS, QUERIES)

    assert captured == [MODEL]
    assert captured[0] is not MODEL
    assert record["model"] == MODEL
    assert record["benchmark"] == {
        "benchmark_id": "healthcare-retrieval",
        "benchmark_version": "v1",
        "top_k": 2,
        "query_count": 2,
    }
    assert record["metrics"] == {
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "precision_at_k": 0.5,
    }
    assert record["qualification"] == {
        "status": "PASS",
        "reason": "QUALIFICATION_THRESHOLDS_SATISFIED",
    }


def test_runner_rejects_dimension_metadata_that_does_not_match_provider():
    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=lambda model: _perfect_provider(),
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    malformed_model = dict(MODEL)
    malformed_model["embedding_dimension"] = 4

    with pytest.raises(
        ValueError,
        match="observed embedding dimension does not match model metadata",
    ):
        runner.qualify(malformed_model, CHUNKS, QUERIES)


def test_runner_rejects_provider_without_embedding_contract():
    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=lambda model: object(),
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    with pytest.raises(
        ValueError,
        match="embedding provider must provide encode_documents",
    ):
        runner.qualify(MODEL, CHUNKS, QUERIES)


def test_runner_rejects_invalid_probe_batch_before_benchmark():
    class InvalidProvider:
        def encode_documents(self, texts):
            return []

    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=lambda model: InvalidProvider(),
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    with pytest.raises(
        ValueError,
        match="embedding provider returned an invalid probe batch",
    ):
        runner.qualify(MODEL, CHUNKS, QUERIES)


def test_runner_validates_model_before_calling_factory():
    calls = []

    def provider_factory(model):
        calls.append(model)
        return _perfect_provider()

    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=provider_factory,
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    malformed_model = dict(MODEL)
    malformed_model["model_revision"] = ""

    with pytest.raises(
        ValueError,
        match="model_revision must be a non-empty string",
    ):
        runner.qualify(malformed_model, CHUNKS, QUERIES)

    assert calls == []


def test_runner_requires_non_empty_chunk_corpus():
    runner = HealthcareEmbeddingQualificationRunner(
        provider_factory=lambda model: _perfect_provider(),
        policy=_policy(),
        benchmark_id="healthcare-retrieval",
        benchmark_version="v1",
        top_k=2,
    )

    with pytest.raises(ValueError, match="chunks must be a non-empty dict"):
        runner.qualify(MODEL, {}, QUERIES)

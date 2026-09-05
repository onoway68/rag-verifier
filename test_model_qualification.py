import pytest

from model_qualification import (
    ModelQualificationPolicy,
    ModelQualificationRecordBuilder
)


def build_policy():
    return ModelQualificationPolicy(
        pass_thresholds={
            "recall_at_k": 0.90,
            "mrr": 0.80
        },
        review_thresholds={
            "recall_at_k": 0.80,
            "mrr": 0.70
        }
    )


def test_policy_passes_when_all_pass_thresholds_met():
    policy = build_policy()

    result = policy.decide({
        "recall_at_k": 0.93,
        "mrr": 0.84,
        "precision_at_k": 0.31
    })

    assert result == {
        "status": "PASS",
        "reason": (
            "QUALIFICATION_THRESHOLDS_SATISFIED"
        )
    }


def test_policy_reviews_when_review_thresholds_met():
    policy = build_policy()

    result = policy.decide({
        "recall_at_k": 0.85,
        "mrr": 0.75,
        "precision_at_k": 0.28
    })

    assert result == {
        "status": "REVIEW",
        "reason": (
            "PASS_THRESHOLDS_NOT_SATISFIED"
        )
    }


def test_policy_fails_below_review_threshold():
    policy = build_policy()

    result = policy.decide({
        "recall_at_k": 0.79,
        "mrr": 0.76,
        "precision_at_k": 0.25
    })

    assert result == {
        "status": "FAIL",
        "reason": (
            "REVIEW_THRESHOLDS_NOT_SATISFIED"
        )
    }


def test_all_configured_pass_metrics_must_pass():
    policy = build_policy()

    result = policy.decide({
        "recall_at_k": 0.95,
        "mrr": 0.79,
        "precision_at_k": 0.40
    })

    assert result["status"] == "REVIEW"


def test_pass_threshold_cannot_be_below_review_threshold():
    with pytest.raises(
        ValueError,
        match=(
            "pass threshold must be greater than "
            "or equal to review threshold"
        )
    ):
        ModelQualificationPolicy(
            pass_thresholds={
                "recall_at_k": 0.70
            },
            review_thresholds={
                "recall_at_k": 0.80
            }
        )


def test_policy_requires_same_metric_keys():
    with pytest.raises(
        ValueError,
        match=(
            "pass and review thresholds must "
            "use the same metrics"
        )
    ):
        ModelQualificationPolicy(
            pass_thresholds={
                "recall_at_k": 0.90
            },
            review_thresholds={
                "mrr": 0.70
            }
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        1.1,
        True,
        "0.9",
        None
    ]
)
def test_invalid_threshold_is_rejected(
    invalid_threshold
):
    with pytest.raises(ValueError):
        ModelQualificationPolicy(
            pass_thresholds={
                "recall_at_k": invalid_threshold
            },
            review_thresholds={
                "recall_at_k": 0.70
            }
        )


def test_missing_required_metric_is_rejected():
    policy = build_policy()

    with pytest.raises(
        ValueError,
        match="qualification metrics are missing"
    ):
        policy.decide({
            "recall_at_k": 0.95
        })


def test_record_binds_model_benchmark_and_metrics():
    policy = build_policy()

    builder = ModelQualificationRecordBuilder(
        policy=policy
    )

    metrics = {
        "query_count": 100,
        "recall_at_k": 0.93,
        "mrr": 0.84,
        "precision_at_k": 0.31
    }

    record = builder.build(
        model={
            "model_id": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "model_revision": "abc123",
            "provider_type": (
                "sentence_transformers"
            ),
            "embedding_dimension": 384
        },
        benchmark={
            "benchmark_id": (
                "healthcare-retrieval"
            ),
            "benchmark_version": "1.0",
            "top_k": 10
        },
        metrics=metrics
    )

    assert record == {
        "model": {
            "model_id": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "model_revision": "abc123",
            "provider_type": (
                "sentence_transformers"
            ),
            "embedding_dimension": 384
        },
        "benchmark": {
            "benchmark_id": (
                "healthcare-retrieval"
            ),
            "benchmark_version": "1.0",
            "top_k": 10,
            "query_count": 100
        },
        "metrics": {
            "recall_at_k": 0.93,
            "mrr": 0.84,
            "precision_at_k": 0.31
        },
        "qualification": {
            "status": "PASS",
            "reason": (
                "QUALIFICATION_THRESHOLDS_SATISFIED"
            )
        }
    }


def test_record_requires_exact_model_revision():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match=(
            "model_revision must be a "
            "non-empty string"
        )
    ):
        builder.build(
            model={
                "model_id": "example/model",
                "model_revision": "",
                "provider_type": "example",
                "embedding_dimension": 384
            },
            benchmark={
                "benchmark_id": "healthcare",
                "benchmark_version": "1.0",
                "top_k": 10
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


def test_record_requires_positive_query_count():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match=(
            "query_count must be a "
            "positive integer"
        )
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
                "top_k": 10
            },
            metrics={
                "query_count": 0,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        None,
        [],
        "invalid"
    ]
)
def test_policy_metrics_must_be_dict(
    invalid_metrics
):
    policy = build_policy()

    with pytest.raises(
        ValueError,
        match=(
            "qualification metrics must be a dict"
        )
    ):
        policy.decide(invalid_metrics)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        True,
        "0.9",
        None
    ]
)
def test_invalid_metric_value_is_rejected(
    invalid_value
):
    policy = build_policy()

    with pytest.raises(
        ValueError,
        match=(
            "qualification metrics must be "
            "numeric values between 0 and 1"
        )
    ):
        policy.decide({
            "recall_at_k": invalid_value,
            "mrr": 0.80
        })


def test_record_rejects_missing_model_metadata():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match="model metadata is incomplete"
    ):
        builder.build(
            model={
                "model_id": "example/model"
            },
            benchmark={
                "benchmark_id": "healthcare",
                "benchmark_version": "1.0",
                "top_k": 10
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


def test_record_rejects_missing_benchmark_metadata():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match="benchmark metadata is incomplete"
    ):
        builder.build(
            model={
                "model_id": "example/model",
                "model_revision": "abc123",
                "provider_type": "example",
                "embedding_dimension": 384
            },
            benchmark={
                "benchmark_id": "healthcare"
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


def test_record_rejects_missing_metrics():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match="metrics are incomplete"
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
                "top_k": 10
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90
            }
        )


def test_record_requires_positive_embedding_dimension():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match=(
            "embedding_dimension must be a "
            "positive integer"
        )
    ):
        builder.build(
            model={
                "model_id": "example/model",
                "model_revision": "abc123",
                "provider_type": "example",
                "embedding_dimension": 0
            },
            benchmark={
                "benchmark_id": "healthcare",
                "benchmark_version": "1.0",
                "top_k": 10
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


def test_record_requires_positive_top_k():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    with pytest.raises(
        ValueError,
        match="top_k must be a positive integer"
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
                "top_k": 0
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


class InvalidQualificationPolicy:
    def decide(self, metrics):
        return {
            "status": "INVALID",
            "reason": ""
        }


def test_record_rejects_invalid_policy_output():
    builder = ModelQualificationRecordBuilder(
        policy=InvalidQualificationPolicy()
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
                "top_k": 10
            },
            metrics={
                "query_count": 10,
                "recall_at_k": 0.90,
                "mrr": 0.80,
                "precision_at_k": 0.20
            }
        )


def test_record_does_not_alias_input_dicts():
    builder = ModelQualificationRecordBuilder(
        policy=build_policy()
    )

    model = {
        "model_id": "example/model",
        "model_revision": "abc123",
        "provider_type": "example",
        "embedding_dimension": 384
    }

    benchmark = {
        "benchmark_id": "healthcare",
        "benchmark_version": "1.0",
        "top_k": 10
    }

    metrics = {
        "query_count": 10,
        "recall_at_k": 0.90,
        "mrr": 0.80,
        "precision_at_k": 0.20
    }

    record = builder.build(
        model=model,
        benchmark=benchmark,
        metrics=metrics
    )

    model["model_id"] = "tampered/model"
    benchmark["benchmark_id"] = "tampered"
    metrics["recall_at_k"] = 0.0

    assert record["model"]["model_id"] == (
        "example/model"
    )

    assert record["benchmark"][
        "benchmark_id"
    ] == "healthcare"

    assert record["metrics"][
        "recall_at_k"
    ] == pytest.approx(0.90)


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf")
    ]
)
def test_non_finite_threshold_is_rejected(
    invalid_threshold
):
    with pytest.raises(ValueError):
        ModelQualificationPolicy(
            pass_thresholds={
                "recall_at_k": invalid_threshold
            },
            review_thresholds={
                "recall_at_k": 0.70
            }
        )


@pytest.mark.parametrize(
    "invalid_metric",
    [
        float("nan"),
        float("inf"),
        float("-inf")
    ]
)
def test_non_finite_metric_is_rejected(
    invalid_metric
):
    policy = build_policy()

    with pytest.raises(ValueError):
        policy.decide({
            "recall_at_k": invalid_metric,
            "mrr": 0.80
        })


def test_record_does_not_alias_policy_output():
    shared_output = {
        "status": "PASS",
        "reason": (
            "QUALIFICATION_THRESHOLDS_SATISFIED"
        )
    }

    class SharedOutputPolicy:
        def decide(self, metrics):
            return shared_output

    builder = ModelQualificationRecordBuilder(
        policy=SharedOutputPolicy()
    )

    record = builder.build(
        model={
            "model_id": "example/model",
            "model_revision": "abc123",
            "provider_type": "example",
            "embedding_dimension": 384
        },
        benchmark={
            "benchmark_id": "healthcare",
            "benchmark_version": "1.0",
            "top_k": 10
        },
        metrics={
            "query_count": 10,
            "recall_at_k": 0.90,
            "mrr": 0.80,
            "precision_at_k": 0.20
        }
    )

    shared_output["status"] = "FAIL"
    shared_output["reason"] = "TAMPERED"

    assert record["qualification"] == {
        "status": "PASS",
        "reason": (
            "QUALIFICATION_THRESHOLDS_SATISFIED"
        )
    }

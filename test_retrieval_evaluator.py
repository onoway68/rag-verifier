import pytest

from retrieval_evaluator import RetrievalEvaluator


def test_precision_at_k_counts_relevant_results():
    evaluator = RetrievalEvaluator()

    score = evaluator.precision_at_k(
        retrieved_ids=[
            "pna-001",
            "mi-001",
            "htn-001"
        ],
        relevant_ids={
            "mi-001"
        },
        k=3
    )

    assert score == pytest.approx(
        1.0 / 3.0
    )


def test_recall_at_k_counts_retrieved_relevant_items():
    evaluator = RetrievalEvaluator()

    score = evaluator.recall_at_k(
        retrieved_ids=[
            "mi-001",
            "pna-001",
            "htn-001"
        ],
        relevant_ids={
            "mi-001",
            "mi-002"
        },
        k=3
    )

    assert score == pytest.approx(
        0.5
    )


def test_reciprocal_rank_uses_first_relevant_result():
    evaluator = RetrievalEvaluator()

    score = evaluator.reciprocal_rank(
        retrieved_ids=[
            "pna-001",
            "mi-001",
            "mi-002"
        ],
        relevant_ids={
            "mi-001",
            "mi-002"
        }
    )

    assert score == pytest.approx(
        0.5
    )


def test_reciprocal_rank_is_one_when_relevant_result_is_first():
    evaluator = RetrievalEvaluator()

    score = evaluator.reciprocal_rank(
        retrieved_ids=[
            "mi-001",
            "pna-001"
        ],
        relevant_ids={
            "mi-001"
        }
    )

    assert score == pytest.approx(
        1.0
    )


def test_reciprocal_rank_is_zero_when_no_relevant_result_is_retrieved():
    evaluator = RetrievalEvaluator()

    score = evaluator.reciprocal_rank(
        retrieved_ids=[
            "pna-001",
            "htn-001"
        ],
        relevant_ids={
            "mi-001"
        }
    )

    assert score == 0.0


def test_precision_at_k_is_zero_for_empty_retrieval():
    evaluator = RetrievalEvaluator()

    score = evaluator.precision_at_k(
        retrieved_ids=[],
        relevant_ids={
            "mi-001"
        },
        k=3
    )

    assert score == 0.0


def test_recall_at_k_is_zero_when_no_relevant_items_are_defined():
    evaluator = RetrievalEvaluator()

    score = evaluator.recall_at_k(
        retrieved_ids=[
            "mi-001"
        ],
        relevant_ids=set(),
        k=3
    )

    assert score == 0.0


@pytest.mark.parametrize(
    "method_name",
    [
        "precision_at_k",
        "recall_at_k"
    ]
)
def test_non_positive_k_is_rejected(
    method_name
):
    evaluator = RetrievalEvaluator()

    method = getattr(
        evaluator,
        method_name
    )

    with pytest.raises(
        ValueError,
        match="k must be greater than zero"
    ):
        method(
            retrieved_ids=[
                "mi-001"
            ],
            relevant_ids={
                "mi-001"
            },
            k=0
        )


def test_evaluate_query_returns_all_metrics():
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate_query(
        retrieved_ids=[
            "pna-001",
            "mi-001",
            "htn-001"
        ],
        relevant_ids={
            "mi-001"
        },
        k=3
    )

    assert result == {
        "precision_at_k": pytest.approx(
            1.0 / 3.0
        ),
        "recall_at_k": pytest.approx(
            1.0
        ),
        "reciprocal_rank": pytest.approx(
            0.5
        )
    }

def test_evaluate_dataset_averages_metrics():
    evaluator = RetrievalEvaluator()

    cases = [
        {
            "retrieved_ids": [
                "mi-001",
                "pna-001",
                "htn-001"
            ],
            "relevant_ids": {
                "mi-001"
            }
        },
        {
            "retrieved_ids": [
                "pna-001",
                "htn-001",
                "mi-001"
            ],
            "relevant_ids": {
                "mi-001"
            }
        }
    ]

    result = evaluator.evaluate_dataset(
        cases=cases,
        k=3
    )

    assert result["mean_precision_at_k"] == pytest.approx(
        1.0 / 3.0
    )

    assert result["mean_recall_at_k"] == pytest.approx(
        1.0
    )

    assert result["mrr"] == pytest.approx(
        (1.0 + (1.0 / 3.0)) / 2.0
    )

    assert result["query_count"] == 2


def test_evaluate_dataset_returns_zero_metrics_for_empty_dataset():
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate_dataset(
        cases=[],
        k=3
    )

    assert result == {
        "mean_precision_at_k": 0.0,
        "mean_recall_at_k": 0.0,
        "mrr": 0.0,
        "query_count": 0
    }

def test_precision_at_k_penalizes_fewer_than_k_results():
    evaluator = RetrievalEvaluator()

    score = evaluator.precision_at_k(
        retrieved_ids=[
            "mi-001"
        ],
        relevant_ids={
            "mi-001"
        },
        k=3
    )

    assert score == pytest.approx(
        1.0 / 3.0
    )


def test_empty_dataset_still_rejects_invalid_k():
    evaluator = RetrievalEvaluator()

    with pytest.raises(
        ValueError,
        match="k must be greater than zero"
    ):
        evaluator.evaluate_dataset(
            cases=[],
            k=0
        )

def test_precision_at_k_does_not_double_count_duplicate_ids():
    evaluator = RetrievalEvaluator()

    score = evaluator.precision_at_k(
        retrieved_ids=[
            "mi-001",
            "mi-001",
            "pna-001"
        ],
        relevant_ids={
            "mi-001"
        },
        k=3
    )

    assert score == pytest.approx(
        1.0 / 3.0
    )

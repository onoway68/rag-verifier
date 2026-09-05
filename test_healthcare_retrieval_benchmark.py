import pytest

from healthcare_retrieval_benchmark import (
    HealthcareRetrievalBenchmark
)


class StaticRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def retrieve(self, query, top_k):
        return list(
            self.results_by_query[query][:top_k]
        )


def test_benchmark_computes_recall_mrr_and_precision():
    queries = [
        {
            "query_id": "Q1",
            "text": (
                "What condition is associated with "
                "persistently elevated blood pressure?"
            ),
            "relevant_ids": ["DOC-HTN"]
        },
        {
            "query_id": "Q2",
            "text": (
                "What disorder is characterized by "
                "elevated blood glucose?"
            ),
            "relevant_ids": ["DOC-DM"]
        }
    ]

    retriever = StaticRetriever({
        queries[0]["text"]: [
            {"chunk_id": "DOC-HTN"},
            {"chunk_id": "DOC-DM"}
        ],
        queries[1]["text"]: [
            {"chunk_id": "DOC-OTHER"},
            {"chunk_id": "DOC-DM"}
        ]
    })

    benchmark = HealthcareRetrievalBenchmark(
        retriever=retriever,
        top_k=2
    )

    result = benchmark.evaluate(queries)

    assert result["query_count"] == 2

    assert result["recall_at_k"] == pytest.approx(
        1.0
    )

    assert result["mrr"] == pytest.approx(
        0.75
    )

    assert result["precision_at_k"] == pytest.approx(
        0.5
    )


def test_benchmark_reports_per_query_results():
    query = {
        "query_id": "Q1",
        "text": "Hypertension question",
        "relevant_ids": ["DOC-HTN"]
    }

    retriever = StaticRetriever({
        query["text"]: [
            {"chunk_id": "DOC-HTN"},
            {"chunk_id": "DOC-DM"}
        ]
    })

    benchmark = HealthcareRetrievalBenchmark(
        retriever=retriever,
        top_k=2
    )

    result = benchmark.evaluate([query])

    assert result["queries"] == [
        {
            "query_id": "Q1",
            "retrieved_ids": [
                "DOC-HTN",
                "DOC-DM"
            ],
            "relevant_ids": [
                "DOC-HTN"
            ],
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "precision_at_k": 0.5
        }
    ]


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        0,
        -1,
        True,
        1.5,
        "2"
    ]
)
def test_invalid_top_k_is_rejected(
    invalid_top_k
):
    with pytest.raises(
        ValueError,
        match="top_k must be a positive integer"
    ):
        HealthcareRetrievalBenchmark(
            retriever=StaticRetriever({}),
            top_k=invalid_top_k
        )


def test_retriever_contract_is_required():
    with pytest.raises(
        ValueError,
        match=(
            "retriever must provide callable "
            r"retrieve\(\)"
        )
    ):
        HealthcareRetrievalBenchmark(
            retriever=object(),
            top_k=2
        )


@pytest.mark.parametrize(
    "queries",
    [
        None,
        {},
        "invalid",
        []
    ]
)
def test_queries_must_be_non_empty_list(
    queries
):
    benchmark = HealthcareRetrievalBenchmark(
        retriever=StaticRetriever({}),
        top_k=2
    )

    with pytest.raises(
        ValueError,
        match="queries must be a non-empty list"
    ):
        benchmark.evaluate(queries)


@pytest.mark.parametrize(
    "query",
    [
        {},
        {
            "query_id": "Q1",
            "text": "",
            "relevant_ids": ["DOC-1"]
        },
        {
            "query_id": "Q1",
            "text": "Question",
            "relevant_ids": []
        },
        {
            "query_id": "",
            "text": "Question",
            "relevant_ids": ["DOC-1"]
        }
    ]
)
def test_invalid_query_contract_is_rejected(query):
    benchmark = HealthcareRetrievalBenchmark(
        retriever=StaticRetriever({}),
        top_k=2
    )

    with pytest.raises(ValueError):
        benchmark.evaluate([query])


def test_duplicate_relevant_ids_are_rejected():
    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": [
            "DOC-1",
            "DOC-1"
        ]
    }

    benchmark = HealthcareRetrievalBenchmark(
        retriever=StaticRetriever({}),
        top_k=2
    )

    with pytest.raises(
        ValueError,
        match="relevant_ids must be unique"
    ):
        benchmark.evaluate([query])


def test_retriever_output_must_be_list():
    class InvalidRetriever:
        def retrieve(self, query, top_k):
            return None

    benchmark = HealthcareRetrievalBenchmark(
        retriever=InvalidRetriever(),
        top_k=2
    )

    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": ["DOC-1"]
    }

    with pytest.raises(
        ValueError,
        match="retriever output must be a list"
    ):
        benchmark.evaluate([query])


def test_retrieved_item_requires_chunk_id():
    class InvalidRetriever:
        def retrieve(self, query, top_k):
            return [{"score": 0.9}]

    benchmark = HealthcareRetrievalBenchmark(
        retriever=InvalidRetriever(),
        top_k=2
    )

    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": ["DOC-1"]
    }

    with pytest.raises(
        ValueError,
        match="retrieved item must contain chunk_id"
    ):
        benchmark.evaluate([query])


def test_duplicate_retrieved_ids_are_rejected():
    retriever = StaticRetriever({
        "Question": [
            {"chunk_id": "DOC-1"},
            {"chunk_id": "DOC-1"}
        ]
    })

    benchmark = HealthcareRetrievalBenchmark(
        retriever=retriever,
        top_k=2
    )

    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": ["DOC-1"]
    }

    with pytest.raises(
        ValueError,
        match="retrieved chunk_ids must be unique"
    ):
        benchmark.evaluate([query])


def test_precision_at_k_uses_configured_k():
    retriever = StaticRetriever({
        "Question": [
            {"chunk_id": "DOC-1"}
        ]
    })

    benchmark = HealthcareRetrievalBenchmark(
        retriever=retriever,
        top_k=2
    )

    query = {
        "query_id": "Q1",
        "text": "Question",
        "relevant_ids": ["DOC-1"]
    }

    result = benchmark.evaluate([query])

    assert result["precision_at_k"] == pytest.approx(
        0.5
    )

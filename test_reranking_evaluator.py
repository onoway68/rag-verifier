import pytest

from embedding_provider import (
    FakeEmbeddingProvider
)
from reranker import Reranker
from reranker_provider import (
    FakeRerankerProvider
)
from reranking_evaluator import (
    evaluate_reranking
)
from retriever import Retriever


def build_system():
    query = "diabetes complications"

    chunks = {
        "hypertension-001": (
            "Hypertension can cause "
            "vascular complications."
        ),
        "diabetes-001": (
            "Diabetes can cause "
            "retinopathy neuropathy "
            "and kidney disease."
        ),
        "pneumonia-001": (
            "Pneumonia is a lung "
            "infection."
        )
    }

    embedding_provider = (
        FakeEmbeddingProvider(
            embeddings={
                query: [
                    1.0,
                    0.0
                ],
                chunks[
                    "hypertension-001"
                ]: [
                    0.99,
                    0.01
                ],
                chunks[
                    "diabetes-001"
                ]: [
                    0.95,
                    0.05
                ],
                chunks[
                    "pneumonia-001"
                ]: [
                    0.0,
                    1.0
                ]
            }
        )
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=(
            embedding_provider
        )
    )

    reranker_provider = (
        FakeRerankerProvider(
            scores={
                (
                    query,
                    chunks[
                        "hypertension-001"
                    ]
                ): 0.30,
                (
                    query,
                    chunks[
                        "diabetes-001"
                    ]
                ): 0.95,
                (
                    query,
                    chunks[
                        "pneumonia-001"
                    ]
                ): 0.05
            }
        )
    )

    reranker = Reranker(
        reranker_provider
    )

    cases = [
        {
            "query": query,
            "relevant_ids": [
                "diabetes-001"
            ]
        }
    ]

    return (
        retriever,
        reranker,
        cases
    )


def test_reranking_improves_rank_one():
    (
        retriever,
        reranker,
        cases
    ) = build_system()

    result = evaluate_reranking(
        retriever=retriever,
        reranker=reranker,
        cases=cases,
        candidate_k=3,
        final_k=1
    )

    assert (
        result["cases"][0][
            "baseline_ids"
        ]
        == ["hypertension-001"]
    )

    assert (
        result["cases"][0][
            "reranked_ids"
        ]
        == ["diabetes-001"]
    )

    assert (
        result["baseline"]["mrr"]
        == 0.0
    )

    assert (
        result["reranked"]["mrr"]
        == 1.0
    )

    assert (
        result["baseline"][
            "mean_recall_at_k"
        ]
        == 0.0
    )

    assert (
        result["reranked"][
            "mean_recall_at_k"
        ]
        == 1.0
    )


def test_reranking_preserves_candidate_provenance():
    (
        retriever,
        reranker,
        cases
    ) = build_system()

    result = evaluate_reranking(
        retriever=retriever,
        reranker=reranker,
        cases=cases,
        candidate_k=3,
        final_k=1
    )

    assert (
        result["cases"][0][
            "candidate_ids"
        ]
        == [
            "hypertension-001",
            "diabetes-001",
            "pneumonia-001"
        ]
    )


@pytest.mark.parametrize(
    "candidate_k",
    [
        0,
        -1,
        True,
        1.5,
        "3"
    ]
)
def test_invalid_candidate_k_is_rejected(
    candidate_k
):
    (
        retriever,
        reranker,
        cases
    ) = build_system()

    with pytest.raises(
        ValueError,
        match=(
            "candidate_k must be "
            "a positive integer"
        )
    ):
        evaluate_reranking(
            retriever=retriever,
            reranker=reranker,
            cases=cases,
            candidate_k=candidate_k,
            final_k=1
        )


@pytest.mark.parametrize(
    "final_k",
    [
        0,
        -1,
        True,
        1.5,
        "1"
    ]
)
def test_invalid_final_k_is_rejected(
    final_k
):
    (
        retriever,
        reranker,
        cases
    ) = build_system()

    with pytest.raises(
        ValueError,
        match=(
            "final_k must be "
            "a positive integer"
        )
    ):
        evaluate_reranking(
            retriever=retriever,
            reranker=reranker,
            cases=cases,
            candidate_k=3,
            final_k=final_k
        )


def test_final_k_cannot_exceed_candidate_k():
    (
        retriever,
        reranker,
        cases
    ) = build_system()

    with pytest.raises(
        ValueError,
        match=(
            "final_k must not exceed "
            "candidate_k"
        )
    ):
        evaluate_reranking(
            retriever=retriever,
            reranker=reranker,
            cases=cases,
            candidate_k=1,
            final_k=2
        )

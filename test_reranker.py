import math

import pytest

from reranker import Reranker
from reranker_provider import (
    FakeRerankerProvider
)


def build_candidates():
    return [
        {
            "chunk_id": "doc-a-000",
            "text": "candidate A",
            "score": 0.90
        },
        {
            "chunk_id": "doc-b-000",
            "text": "candidate B",
            "score": 0.80
        },
        {
            "chunk_id": "doc-c-000",
            "text": "candidate C",
            "score": 0.70
        }
    ]


def test_reranker_reorders_candidates():
    query = "test query"

    provider = FakeRerankerProvider(
        scores={
            (
                query,
                "candidate A"
            ): 0.20,
            (
                query,
                "candidate B"
            ): 0.95,
            (
                query,
                "candidate C"
            ): 0.40
        }
    )

    reranker = Reranker(
        reranker_provider=provider
    )

    results = reranker.rerank(
        query=query,
        candidates=build_candidates()
    )

    assert [
        item["chunk_id"]
        for item in results
    ] == [
        "doc-b-000",
        "doc-c-000",
        "doc-a-000"
    ]


def test_reranker_preserves_retrieval_score():
    query = "test query"

    provider = FakeRerankerProvider(
        default_score=0.5
    )

    reranker = Reranker(provider)

    results = reranker.rerank(
        query,
        build_candidates()
    )

    scores_by_id = {
        item["chunk_id"]: (
            item["retrieval_score"]
        )
        for item in results
    }

    assert scores_by_id == {
        "doc-a-000": 0.90,
        "doc-b-000": 0.80,
        "doc-c-000": 0.70
    }


def test_top_k_limits_reranked_results():
    query = "test query"

    provider = FakeRerankerProvider(
        scores={
            (
                query,
                "candidate A"
            ): 0.1,
            (
                query,
                "candidate B"
            ): 0.9,
            (
                query,
                "candidate C"
            ): 0.8
        }
    )

    reranker = Reranker(provider)

    results = reranker.rerank(
        query=query,
        candidates=build_candidates(),
        top_k=2
    )

    assert len(results) == 2

    assert [
        item["chunk_id"]
        for item in results
    ] == [
        "doc-b-000",
        "doc-c-000"
    ]


def test_empty_candidates_return_empty_list():
    provider = FakeRerankerProvider()

    reranker = Reranker(provider)

    assert (
        reranker.rerank(
            "query",
            []
        )
        == []
    )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        True,
        1.5,
        "2"
    ]
)
def test_invalid_top_k_is_rejected(
    top_k
):
    provider = FakeRerankerProvider()

    reranker = Reranker(provider)

    with pytest.raises(
        ValueError,
        match=(
            "top_k must be a positive integer"
        )
    ):
        reranker.rerank(
            "query",
            build_candidates(),
            top_k=top_k
        )


def test_nonlist_candidates_are_rejected():
    reranker = Reranker(
        FakeRerankerProvider()
    )

    with pytest.raises(
        ValueError,
        match="candidates must be a list"
    ):
        reranker.rerank(
            "query",
            {}
        )


@pytest.mark.parametrize(
    "candidate, message",
    [
        (
            "not-a-dict",
            (
                "Each candidate must "
                "be a dictionary"
            )
        ),
        (
            {
                "text": "candidate",
                "score": 0.5
            },
            "Candidate is missing chunk_id"
        ),
        (
            {
                "chunk_id": "doc-001",
                "score": 0.5
            },
            "Candidate is missing text"
        ),
        (
            {
                "chunk_id": "doc-001",
                "text": "candidate"
            },
            (
                "Candidate is missing "
                "retrieval score"
            )
        )
    ]
)
def test_malformed_candidate_is_rejected(
    candidate,
    message
):
    reranker = Reranker(
        FakeRerankerProvider()
    )

    with pytest.raises(
        ValueError,
        match=message
    ):
        reranker.rerank(
            "query",
            [candidate]
        )


def test_provider_wrong_score_count_is_rejected():
    class BadProvider:
        def score(
            self,
            query,
            texts
        ):
            return [0.5]

    reranker = Reranker(
        BadProvider()
    )

    with pytest.raises(
        ValueError,
        match=(
            "unexpected number of scores"
        )
    ):
        reranker.rerank(
            "query",
            build_candidates()
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        "0.5",
        None,
        True
    ]
)
def test_nonnumeric_score_is_rejected(
    invalid_score
):
    class BadProvider:
        def score(
            self,
            query,
            texts
        ):
            return [
                invalid_score
                for _ in texts
            ]

    reranker = Reranker(
        BadProvider()
    )

    with pytest.raises(
        ValueError,
        match=(
            "Reranker scores must be numeric"
        )
    ):
        reranker.rerank(
            "query",
            build_candidates()
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        math.nan,
        math.inf,
        -math.inf
    ]
)
def test_nonfinite_score_is_rejected(
    invalid_score
):
    class BadProvider:
        def score(
            self,
            query,
            texts
        ):
            return [
                invalid_score
                for _ in texts
            ]

    reranker = Reranker(
        BadProvider()
    )

    with pytest.raises(
        ValueError,
        match=(
            "Reranker scores must be finite"
        )
    ):
        reranker.rerank(
            "query",
            build_candidates()
        )

import math

import pytest

from context_builder import ContextBuilder


def make_evidence(
    chunk_id="chunk-001",
    text="alpha beta gamma",
    retrieval_score=0.8,
    rerank_score=2.5
):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "retrieval_score": retrieval_score,
        "rerank_score": rerank_score
    }


def test_build_preserves_evidence_order():
    builder = ContextBuilder(max_words=100)

    evidence = [
        make_evidence(
            chunk_id="chunk-002",
            text="second ranked evidence",
            rerank_score=9.0
        ),
        make_evidence(
            chunk_id="chunk-001",
            text="first original evidence",
            rerank_score=1.0
        )
    ]

    result = builder.build(evidence)

    assert [
        item["chunk_id"]
        for item in result["evidence"]
    ] == [
        "chunk-002",
        "chunk-001"
    ]


def test_build_assigns_stable_citation_ids():
    builder = ContextBuilder(max_words=100)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001"
            ),
            make_evidence(
                chunk_id="chunk-002"
            )
        ]
    )

    assert [
        item["citation_id"]
        for item in result["evidence"]
    ] == [
        "C1",
        "C2"
    ]


def test_build_preserves_provenance():
    builder = ContextBuilder(max_words=100)

    result = builder.build(
        [
            make_evidence(
                chunk_id="diabetes-001",
                text="diabetes complication evidence",
                retrieval_score=0.558774,
                rerank_score=7.89832
            )
        ]
    )

    item = result["evidence"][0]

    assert item["chunk_id"] == "diabetes-001"
    assert item["retrieval_score"] == pytest.approx(
        0.558774
    )
    assert item["rerank_score"] == pytest.approx(
        7.89832
    )


def test_build_constructs_citation_aware_context():
    builder = ContextBuilder(max_words=100)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text="alpha beta"
            ),
            make_evidence(
                chunk_id="chunk-002",
                text="gamma delta"
            )
        ]
    )

    assert result["context"] == (
        "[C1] alpha beta\n\n"
        "[C2] gamma delta"
    )


def test_build_reports_word_count():
    builder = ContextBuilder(max_words=100)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text="one two three"
            ),
            make_evidence(
                chunk_id="chunk-002",
                text="four five"
            )
        ]
    )

    assert result["word_count"] == 5
    assert result["max_words"] == 100


def test_build_respects_exact_budget():
    builder = ContextBuilder(max_words=5)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text="one two three"
            ),
            make_evidence(
                chunk_id="chunk-002",
                text="four five"
            )
        ]
    )

    assert [
        item["chunk_id"]
        for item in result["evidence"]
    ] == [
        "chunk-001",
        "chunk-002"
    ]

    assert result["word_count"] == 5


def test_build_skips_item_that_exceeds_remaining_budget():
    builder = ContextBuilder(max_words=5)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text="one two three"
            ),
            make_evidence(
                chunk_id="chunk-002",
                text="four five six"
            ),
            make_evidence(
                chunk_id="chunk-003",
                text="seven eight"
            )
        ]
    )

    assert [
        item["chunk_id"]
        for item in result["evidence"]
    ] == [
        "chunk-001",
        "chunk-003"
    ]

    assert result["word_count"] == 5


def test_skipped_item_does_not_consume_citation_id():
    builder = ContextBuilder(max_words=5)

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text="one two three"
            ),
            make_evidence(
                chunk_id="chunk-002",
                text="four five six"
            ),
            make_evidence(
                chunk_id="chunk-003",
                text="seven eight"
            )
        ]
    )

    assert [
        item["citation_id"]
        for item in result["evidence"]
    ] == [
        "C1",
        "C2"
    ]


def test_empty_evidence_produces_empty_context():
    builder = ContextBuilder(max_words=100)

    result = builder.build([])

    assert result == {
        "context": "",
        "evidence": [],
        "citation_map": {},
        "word_count": 0,
        "max_words": 100
    }


@pytest.mark.parametrize(
    "max_words",
    [
        0,
        -1,
        True,
        1.5,
        "10"
    ]
)
def test_invalid_max_words_rejected(
    max_words
):
    with pytest.raises(
        ValueError,
        match=(
            "max_words must be a positive integer"
        )
    ):
        ContextBuilder(
            max_words=max_words
        )


def test_nonlist_evidence_rejected():
    builder = ContextBuilder()

    with pytest.raises(
        ValueError,
        match="evidence must be a list"
    ):
        builder.build(
            {"chunk_id": "chunk-001"}
        )


@pytest.mark.parametrize(
    "item",
    [
        "not-a-dictionary",
        123,
        None
    ]
)
def test_nondictionary_evidence_item_rejected(
    item
):
    builder = ContextBuilder()

    with pytest.raises(
        ValueError,
        match=(
            "Each evidence item must be a dictionary"
        )
    ):
        builder.build([item])


@pytest.mark.parametrize(
    "missing_field",
    [
        "chunk_id",
        "text",
        "retrieval_score",
        "rerank_score"
    ]
)
def test_missing_required_field_rejected(
    missing_field
):
    builder = ContextBuilder()

    item = make_evidence()

    del item[missing_field]

    with pytest.raises(
        ValueError,
        match=(
            "Evidence item is missing required fields"
        )
    ):
        builder.build([item])


@pytest.mark.parametrize(
    "chunk_id",
    [
        "",
        "   ",
        None,
        123
    ]
)
def test_invalid_chunk_id_rejected(
    chunk_id
):
    builder = ContextBuilder()

    with pytest.raises(
        ValueError,
        match=(
            "chunk_id must be a non-empty string"
        )
    ):
        builder.build(
            [
                make_evidence(
                    chunk_id=chunk_id
                )
            ]
        )


def test_duplicate_chunk_id_rejected():
    builder = ContextBuilder()

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate chunk_id is not allowed"
        )
    ):
        builder.build(
            [
                make_evidence(
                    chunk_id="chunk-001"
                ),
                make_evidence(
                    chunk_id="chunk-001"
                )
            ]
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        None,
        123
    ]
)
def test_invalid_text_rejected(
    text
):
    builder = ContextBuilder()

    with pytest.raises(
        ValueError,
        match=(
            "text must be a non-empty string"
        )
    ):
        builder.build(
            [
                make_evidence(
                    text=text
                )
            ]
        )


@pytest.mark.parametrize(
    "score_name",
    [
        "retrieval_score",
        "rerank_score"
    ]
)
@pytest.mark.parametrize(
    "bad_score",
    [
        "0.5",
        None,
        True
    ]
)
def test_nonnumeric_score_rejected(
    score_name,
    bad_score
):
    builder = ContextBuilder()

    item = make_evidence()

    item[score_name] = bad_score

    with pytest.raises(
        ValueError,
        match=(
            f"{score_name} must be numeric"
        )
    ):
        builder.build([item])


@pytest.mark.parametrize(
    "score_name",
    [
        "retrieval_score",
        "rerank_score"
    ]
)
@pytest.mark.parametrize(
    "bad_score",
    [
        math.nan,
        math.inf,
        -math.inf
    ]
)
def test_nonfinite_score_rejected(
    score_name,
    bad_score
):
    builder = ContextBuilder()

    item = make_evidence()

    item[score_name] = bad_score

    with pytest.raises(
        ValueError,
        match=(
            f"{score_name} must be finite"
        )
    ):
        builder.build([item])


def test_oversized_evidence_is_not_truncated():
    builder = ContextBuilder(
        max_words=3
    )

    result = builder.build(
        [
            make_evidence(
                chunk_id="chunk-001",
                text=(
                    "one two three four five"
                )
            )
        ]
    )

    assert result["context"] == ""
    assert result["evidence"] == []
    assert result["word_count"] == 0
    assert result["max_words"] == 3


def test_build_returns_citation_map():
    builder = ContextBuilder(
        max_words=100
    )

    result = builder.build(
        [
            make_evidence(
                chunk_id="diabetes-001",
                text=(
                    "Diabetes can cause "
                    "retinopathy and neuropathy."
                )
            ),
            make_evidence(
                chunk_id="hypertension-001",
                text=(
                    "Hypertension can cause "
                    "cardiovascular complications."
                )
            )
        ]
    )

    assert result["citation_map"] == {
        "C1": (
            "Diabetes can cause "
            "retinopathy and neuropathy."
        ),
        "C2": (
            "Hypertension can cause "
            "cardiovascular complications."
        )
    }

    empty_result = ContextBuilder(
        max_words=2
    ).build(
        [
            make_evidence(
                chunk_id="oversized-001",
                text="one two three"
            )
        ]
    )

    assert empty_result["citation_map"] == {}

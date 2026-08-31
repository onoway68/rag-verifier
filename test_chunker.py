import pytest

from chunker import WordChunker


def test_chunker_splits_text_by_word_count():
    chunker = WordChunker(
        chunk_size=4,
        overlap=0
    )

    chunks = chunker.chunk(
        "A B C D E F G H I",
        document_id="doc1"
    )

    assert [
        chunk["text"]
        for chunk in chunks
    ] == [
        "A B C D",
        "E F G H",
        "I"
    ]


def test_chunker_preserves_configured_overlap():
    chunker = WordChunker(
        chunk_size=4,
        overlap=1
    )

    chunks = chunker.chunk(
        "A B C D E F G H I J",
        document_id="doc1"
    )

    assert [
        chunk["text"]
        for chunk in chunks
    ] == [
        "A B C D",
        "D E F G",
        "G H I J"
    ]


def test_chunk_ids_are_stable_and_document_scoped():
    chunker = WordChunker(
        chunk_size=3
    )

    chunks = chunker.chunk(
        "A B C D E F G",
        document_id="patient-guide"
    )

    assert [
        chunk["chunk_id"]
        for chunk in chunks
    ] == [
        "patient-guide-000",
        "patient-guide-001",
        "patient-guide-002"
    ]


def test_chunker_records_word_offsets():
    chunker = WordChunker(
        chunk_size=4,
        overlap=1
    )

    chunks = chunker.chunk(
        "A B C D E F G H I J",
        document_id="doc1"
    )

    assert [
        (
            chunk["start_word"],
            chunk["end_word"]
        )
        for chunk in chunks
    ] == [
        (0, 4),
        (3, 7),
        (6, 10)
    ]


def test_empty_text_returns_no_chunks():
    chunker = WordChunker(
        chunk_size=4
    )

    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


@pytest.mark.parametrize(
    "chunk_size, overlap, message",
    [
        (
            0,
            0,
            "chunk_size must be greater than zero"
        ),
        (
            -1,
            0,
            "chunk_size must be greater than zero"
        ),
        (
            4,
            -1,
            "overlap must not be negative"
        ),
        (
            4,
            4,
            "overlap must be smaller than chunk_size"
        ),
        (
            4,
            5,
            "overlap must be smaller than chunk_size"
        )
    ]
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size,
    overlap,
    message
):
    with pytest.raises(
        ValueError,
        match=message
    ):
        WordChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )

def test_chunk_size_rejects_boolean():
    with pytest.raises(
        ValueError,
        match="chunk_size must be an integer"
    ):
        WordChunker(
            chunk_size=True
        )


def test_chunk_size_rejects_float():
    with pytest.raises(
        ValueError,
        match="chunk_size must be an integer"
    ):
        WordChunker(
            chunk_size=40.5
        )


def test_chunk_size_rejects_string():
    with pytest.raises(
        ValueError,
        match="chunk_size must be an integer"
    ):
        WordChunker(
            chunk_size="40"
        )


def test_overlap_rejects_boolean():
    with pytest.raises(
        ValueError,
        match="overlap must be an integer"
    ):
        WordChunker(
            chunk_size=40,
            overlap=True
        )


def test_overlap_rejects_float():
    with pytest.raises(
        ValueError,
        match="overlap must be an integer"
    ):
        WordChunker(
            chunk_size=40,
            overlap=10.5
        )


def test_overlap_rejects_string():
    with pytest.raises(
        ValueError,
        match="overlap must be an integer"
    ):
        WordChunker(
            chunk_size=40,
            overlap="10"
        )

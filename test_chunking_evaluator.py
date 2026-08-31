import pytest

from chunker import WordChunker
from chunking_evaluator import (
    build_chunk_corpus,
    chunk_id_to_document_id,
    evaluate_chunking_strategy
)
from embedding_provider import FakeEmbeddingProvider


def test_build_chunk_corpus_flattens_document_chunks():
    documents = {
        "doc1": "A B C D E",
        "doc2": "F G H I"
    }

    chunker = WordChunker(
        chunk_size=3,
        overlap=0
    )

    chunks = build_chunk_corpus(
        documents,
        chunker
    )

    assert chunks == {
        "doc1-000": "A B C",
        "doc1-001": "D E",
        "doc2-000": "F G H",
        "doc2-001": "I"
    }


def test_chunk_id_maps_back_to_document_id():
    assert (
        chunk_id_to_document_id(
            "heart-failure-guide-007"
        )
        == "heart-failure-guide"
    )


@pytest.mark.parametrize(
    "chunk_id",
    [
        "doc",
        "-001",
        "doc-abc"
    ]
)
def test_invalid_chunk_id_is_rejected(
    chunk_id
):
    with pytest.raises(
        ValueError,
        match="Invalid chunk_id format"
    ):
        chunk_id_to_document_id(
            chunk_id
        )


def test_chunking_strategy_evaluates_document_relevance():
    documents = {
        "doc1": "heart muscle disease",
        "doc2": "lung infection pneumonia"
    }

    chunker = WordChunker(
        chunk_size=3,
        overlap=0
    )

    query = "heart disease"

    provider = FakeEmbeddingProvider(
        embeddings={
            query: [1.0, 0.0],
            "heart muscle disease": [
                1.0,
                0.0
            ],
            "lung infection pneumonia": [
                0.0,
                1.0
            ]
        }
    )

    queries = [
        {
            "query": query,
            "relevant_document_ids": [
                "doc1"
            ]
        }
    ]

    result = evaluate_chunking_strategy(
        documents=documents,
        queries=queries,
        chunker=chunker,
        embedding_provider=provider,
        top_k=1
    )

    assert result["chunk_count"] == 2

    assert (
        result["summary"]["query_count"]
        == 1
    )

    assert (
        result["summary"][
            "mean_recall_at_k"
        ]
        == 1.0
    )

    assert result["summary"]["mrr"] == 1.0

    assert (
        result["queries"][0][
            "retrieved_chunk_ids"
        ]
        == ["doc1-000"]
    )

    assert (
        result["queries"][0][
            "retrieved_document_ids"
        ]
        == ["doc1"]
    )

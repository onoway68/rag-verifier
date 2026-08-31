from context_builder import ContextBuilder
from embedding_provider import FakeEmbeddingProvider
from reranker import Reranker
from reranker_provider import FakeRerankerProvider
from retriever import Retriever


def test_retrieval_reranking_context_pipeline():
    query = (
        "What disease can cause "
        "retinopathy and neuropathy?"
    )

    hypertension_text = (
        "Hypertension can cause "
        "cardiovascular complications."
    )

    diabetes_text = (
        "Diabetes can cause retinopathy "
        "and neuropathy."
    )

    chunks = {
        "hypertension-001": hypertension_text,
        "diabetes-001": diabetes_text
    }

    embedding_provider = FakeEmbeddingProvider(
        {
            query: [1.0, 0.0],
            hypertension_text: [1.0, 0.0],
            diabetes_text: [0.8, 0.6]
        }
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=embedding_provider
    )

    retrieved = retriever.retrieve(
        query=query,
        top_k=2
    )

    assert [
        item["chunk_id"]
        for item in retrieved
    ] == [
        "hypertension-001",
        "diabetes-001"
    ]

    reranker_provider = FakeRerankerProvider()

    reranker_provider.set_score(
        query,
        hypertension_text,
        1.0
    )

    reranker_provider.set_score(
        query,
        diabetes_text,
        9.0
    )

    reranker = Reranker(
        reranker_provider=reranker_provider
    )

    reranked = reranker.rerank(
        query=query,
        candidates=retrieved,
        top_k=2
    )

    assert [
        item["chunk_id"]
        for item in reranked
    ] == [
        "diabetes-001",
        "hypertension-001"
    ]

    builder = ContextBuilder(
        max_words=100
    )

    result = builder.build(
        reranked
    )

    assert [
        item["chunk_id"]
        for item in result["evidence"]
    ] == [
        "diabetes-001",
        "hypertension-001"
    ]

    assert [
        item["citation_id"]
        for item in result["evidence"]
    ] == [
        "C1",
        "C2"
    ]

    assert result["context"] == (
        "[C1] "
        + diabetes_text
        + "\n\n[C2] "
        + hypertension_text
    )

    assert result["citation_map"] == {
        "C1": diabetes_text,
        "C2": hypertension_text
    }

    diabetes = result["evidence"][0]

    assert (
        diabetes["retrieval_score"]
        == retrieved[1]["score"]
    )

    assert (
        diabetes["rerank_score"]
        == 9.0
    )

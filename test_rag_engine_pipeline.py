from context_builder import ContextBuilder
from embedding_provider import FakeEmbeddingProvider
from generator import CitationAwareGenerator
from generator_provider import FakeGeneratorProvider
from reranker import Reranker
from reranker_provider import FakeRerankerProvider
from retriever import Retriever


def test_deterministic_rag_engine_pipeline():
    chunks = {
        "diabetes-001": (
            "Diabetes can cause peripheral neuropathy."
        ),
        "hypertension-001": (
            "Hypertension increases cardiovascular risk."
        ),
        "asthma-001": (
            "Asthma is a chronic inflammatory airway disease."
        )
    }

    embeddings = {
        "What complications are described?": [
            1.0,
            0.0
        ],
        chunks["diabetes-001"]: [
            0.9,
            0.1
        ],
        chunks["hypertension-001"]: [
            0.8,
            0.2
        ],
        chunks["asthma-001"]: [
            0.1,
            0.9
        ]
    }

    embedding_provider = FakeEmbeddingProvider(
        embeddings=embeddings
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=embedding_provider
    )

    retrieved = retriever.retrieve(
        "What complications are described?",
        top_k=3
    )

    reranker_provider = FakeRerankerProvider(
        scores={
            (
                "What complications are described?",
                chunks["diabetes-001"]
            ): 9.0,
            (
                "What complications are described?",
                chunks["hypertension-001"]
            ): 8.0,
            (
                "What complications are described?",
                chunks["asthma-001"]
            ): 1.0
        }
    )

    reranker = Reranker(
        reranker_provider=reranker_provider
    )

    reranked = reranker.rerank(
        query="What complications are described?",
        candidates=retrieved,
        top_k=2
    )

    context_builder = ContextBuilder(
        max_words=100
    )

    context_result = context_builder.build(
        reranked
    )

    generator_provider = FakeGeneratorProvider(
        response=(
            "Diabetes can cause peripheral "
            "neuropathy [C1]. "
            "Hypertension increases "
            "cardiovascular risk [C2]."
        )
    )

    generator = CitationAwareGenerator(
        generator_provider
    )

    result = generator.generate(
        question=(
            "What complications are described?"
        ),
        context=context_result["context"],
        citation_map=(
            context_result["citation_map"]
        )
    )

    assert [
        item["chunk_id"]
        for item in reranked
    ] == [
        "diabetes-001",
        "hypertension-001"
    ]

    assert context_result["citation_map"] == {
        "C1": (
            "Diabetes can cause peripheral neuropathy."
        ),
        "C2": (
            "Hypertension increases cardiovascular risk."
        )
    }

    assert result["citation_ids"] == [
        "C1",
        "C2"
    ]

    assert result["answer"] == (
        "Diabetes can cause peripheral "
        "neuropathy [C1]. "
        "Hypertension increases "
        "cardiovascular risk [C2]."
    )

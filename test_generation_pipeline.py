from context_builder import ContextBuilder
from generator import CitationAwareGenerator
from generator_provider import FakeGeneratorProvider


def test_context_builder_output_flows_into_generator():
    ranked_evidence = [
        {
            "chunk_id": "diabetes-001",
            "text": (
                "Diabetes can cause peripheral neuropathy."
            ),
            "retrieval_score": 0.91,
            "rerank_score": 8.4
        },
        {
            "chunk_id": "hypertension-001",
            "text": (
                "Hypertension increases cardiovascular risk."
            ),
            "retrieval_score": 0.84,
            "rerank_score": 7.6
        }
    ]

    context_builder = ContextBuilder(
        max_words=100
    )

    context_result = context_builder.build(
        ranked_evidence
    )

    provider = FakeGeneratorProvider(
        response=(
            "Diabetes can cause peripheral "
            "neuropathy [C1]. "
            "Hypertension increases "
            "cardiovascular risk [C2]."
        )
    )

    generator = CitationAwareGenerator(
        provider
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

    assert result["citation_ids"] == [
        "C1",
        "C2"
    ]

    assert (
        context_result["citation_map"]["C1"]
        ==
        "Diabetes can cause peripheral neuropathy."
    )

    assert (
        context_result["citation_map"]["C2"]
        ==
        "Hypertension increases cardiovascular risk."
    )


def test_budgeted_context_preserves_generator_contract():
    ranked_evidence = [
        {
            "chunk_id": "chunk-001",
            "text": "one two three four",
            "retrieval_score": 0.9,
            "rerank_score": 9.0
        },
        {
            "chunk_id": "chunk-002",
            "text": "five six seven eight",
            "retrieval_score": 0.8,
            "rerank_score": 8.0
        },
        {
            "chunk_id": "chunk-003",
            "text": "nine ten",
            "retrieval_score": 0.7,
            "rerank_score": 7.0
        }
    ]

    context_builder = ContextBuilder(
        max_words=6
    )

    context_result = context_builder.build(
        ranked_evidence
    )

    assert list(
        context_result["citation_map"].keys()
    ) == [
        "C1",
        "C2"
    ]

    provider = FakeGeneratorProvider(
        response="Supported by evidence [C1] [C2]."
    )

    generator = CitationAwareGenerator(
        provider
    )

    result = generator.generate(
        question="What does the evidence say?",
        context=context_result["context"],
        citation_map=(
            context_result["citation_map"]
        )
    )

    assert result["citation_ids"] == [
        "C1",
        "C2"
    ]

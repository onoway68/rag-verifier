import pytest
from context_builder import ContextBuilder
from embedding_provider import FakeEmbeddingProvider
from generator import CitationAwareGenerator
from generator_provider import FakeGeneratorProvider
from nli_provider import FakeNLIProvider
from reranker import Reranker
from reranker_provider import FakeRerankerProvider
from retriever import Retriever
from self_verifying_rag import SelfVerifyingRAG


def build_pipeline(answer):
    question = (
        "What condition is associated with "
        "persistently elevated blood pressure?"
    )

    chunks = {
        "hypertension-001": (
            "Hypertension is a condition characterized "
            "by persistently elevated blood pressure."
        ),
        "diabetes-001": (
            "Diabetes mellitus is characterized by "
            "elevated blood glucose."
        )
    }

    embedding_provider = FakeEmbeddingProvider(
        embeddings={
            question: [1.0, 0.0],
            chunks["hypertension-001"]: [
                1.0,
                0.0
            ],
            chunks["diabetes-001"]: [
                0.0,
                1.0
            ]
        }
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=embedding_provider
    )

    reranker_provider = FakeRerankerProvider(
        scores={
            (
                question,
                chunks["hypertension-001"]
            ): 10.0,
            (
                question,
                chunks["diabetes-001"]
            ): 1.0
        }
    )

    reranker = Reranker(
        reranker_provider=reranker_provider
    )

    context_builder = ContextBuilder(
        max_words=100
    )

    generator = CitationAwareGenerator(
        provider=FakeGeneratorProvider(
            response=answer
        )
    )

    nli_provider = FakeNLIProvider()

    pipeline = SelfVerifyingRAG(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        generator=generator,
        nli_provider=nli_provider,
        retrieval_k=2,
        rerank_k=1
    )

    return (
        pipeline,
        question,
        chunks
    )


def test_self_verifying_rag_passes_supported_answer():
    pipeline, question, chunks = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    result = pipeline.run(
        question
    )

    assert result["question"] == question

    assert (
        result["retrieval"][0]["chunk_id"]
        == "hypertension-001"
    )

    assert (
        result[
            "reranked_evidence"
        ][0]["chunk_id"]
        == "hypertension-001"
    )

    assert (
        result["context"]["citation_map"]
        == {
            "C1": chunks[
                "hypertension-001"
            ]
        }
    )

    assert result["generation"] == {
        "answer": (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        ),
        "citation_ids": ["C1"]
    }

    assert (
        result["verification"]["status"]
        == "PASS"
    )

    assert (
        result["verification"][
            "trusted_release"
        ]
        is True
    )

    assert (
        result["trusted_release"]
        is True
    )


def test_self_verifying_rag_fails_uncited_answer():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure."
        )
    )

    result = pipeline.run(
        question
    )

    assert (
        result["verification"]["status"]
        == "FAIL"
    )

    assert (
        result["verification"]["reason"]
        == "ONE_OR_MORE_CLAIMS_FAILED"
    )

    assert (
        result["verification"]["claims"][0][
            "reason"
        ]
        == "UNCITED_FACTUAL_CLAIM"
    )

    assert (
        result["trusted_release"]
        is False
    )


def test_self_verifying_rag_fails_contradicted_answer():
    pipeline, question, chunks = build_pipeline(
        (
            "Hypertension is not associated with "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.nli_provider.set_response(
        chunks["hypertension-001"],
        (
            "Hypertension is not associated with "
            "persistently elevated blood pressure."
        ),
        {
            "contradiction": 0.98,
            "entailment": 0.01,
            "neutral": 0.01
        }
    )

    result = pipeline.run(
        question
    )

    assert (
        result["verification"]["status"]
        == "FAIL"
    )

    assert (
        result["verification"]["claims"][0][
            "verification"
        ][0]["reason"]
        == "CONTRADICTION_THRESHOLD_MET"
    )

    assert (
        result["trusted_release"]
        is False
    )


def test_invalid_retrieval_k_is_rejected():
    pipeline, _, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    invalid_values = [
        0,
        -1,
        True,
        1.5,
        "2"
    ]

    for value in invalid_values:
        try:
            SelfVerifyingRAG(
                retriever=pipeline.retriever,
                reranker=pipeline.reranker,
                context_builder=pipeline.context_builder,
                generator=pipeline.generator,
                nli_provider=pipeline.nli_provider,
                retrieval_k=value,
                rerank_k=1
            )
        except ValueError as error:
            assert (
                str(error)
                == "retrieval_k must be a positive integer"
            )
        else:
            raise AssertionError(
                f"retrieval_k={value!r} was not rejected"
            )


def test_invalid_rerank_k_is_rejected():
    pipeline, _, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    invalid_values = [
        0,
        -1,
        True,
        1.5,
        "2"
    ]

    for value in invalid_values:
        try:
            SelfVerifyingRAG(
                retriever=pipeline.retriever,
                reranker=pipeline.reranker,
                context_builder=pipeline.context_builder,
                generator=pipeline.generator,
                nli_provider=pipeline.nli_provider,
                retrieval_k=5,
                rerank_k=value
            )
        except ValueError as error:
            assert (
                str(error)
                == "rerank_k must be a positive integer"
            )
        else:
            raise AssertionError(
                f"rerank_k={value!r} was not rejected"
            )


def test_rerank_k_cannot_exceed_retrieval_k():
    pipeline, _, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    try:
        SelfVerifyingRAG(
            retriever=pipeline.retriever,
            reranker=pipeline.reranker,
            context_builder=pipeline.context_builder,
            generator=pipeline.generator,
            nli_provider=pipeline.nli_provider,
            retrieval_k=2,
            rerank_k=3
        )
    except ValueError as error:
        assert (
            str(error)
            == "rerank_k must not exceed retrieval_k"
        )
    else:
        raise AssertionError(
            "rerank_k > retrieval_k was not rejected"
        )



def test_none_dependencies_are_rejected():
    pipeline, _, _ = build_pipeline(
        "Hypertension is associated with persistently "
        "elevated blood pressure [C1]."
    )

    dependencies = (
        "retriever",
        "reranker",
        "context_builder",
        "generator",
        "nli_provider"
    )

    for dependency_name in dependencies:
        kwargs = {
            "retriever": pipeline.retriever,
            "reranker": pipeline.reranker,
            "context_builder": pipeline.context_builder,
            "generator": pipeline.generator,
            "nli_provider": pipeline.nli_provider
        }

        kwargs[dependency_name] = None

        with pytest.raises(
            ValueError,
            match=(
                f"{dependency_name} must not be None"
            )
        ):
            SelfVerifyingRAG(
                **kwargs,
                retrieval_k=2,
                rerank_k=1
            )


def test_dependencies_must_provide_required_methods():
    pipeline, _, _ = build_pipeline(
        "Hypertension is associated with persistently "
        "elevated blood pressure [C1]."
    )

    dependency_contracts = (
        ("retriever", "retrieve"),
        ("reranker", "rerank"),
        ("context_builder", "build"),
        ("generator", "generate"),
        ("nli_provider", "predict")
    )

    for dependency_name, method_name in dependency_contracts:
        kwargs = {
            "retriever": pipeline.retriever,
            "reranker": pipeline.reranker,
            "context_builder": pipeline.context_builder,
            "generator": pipeline.generator,
            "nli_provider": pipeline.nli_provider
        }

        kwargs[dependency_name] = object()

        with pytest.raises(
            ValueError,
            match=(
                f"{dependency_name} must provide callable "
                rf"{method_name}\(\)"
            )
        ):
            SelfVerifyingRAG(
                **kwargs,
                retrieval_k=2,
                rerank_k=1
            )


@pytest.mark.parametrize(
    "threshold_name",
    [
        "pass_threshold",
        "fail_threshold"
    ]
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        False,
        "0.9",
        None
    ]
)
def test_non_numeric_thresholds_are_rejected(
    threshold_name,
    invalid_value
):
    pipeline, _, _ = build_pipeline(
        "Hypertension is associated with persistently "
        "elevated blood pressure [C1]."
    )

    kwargs = {
        "retriever": pipeline.retriever,
        "reranker": pipeline.reranker,
        "context_builder": pipeline.context_builder,
        "generator": pipeline.generator,
        "nli_provider": pipeline.nli_provider,
        "retrieval_k": 2,
        "rerank_k": 1
    }

    kwargs[threshold_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{threshold_name} must be numeric"
    ):
        SelfVerifyingRAG(
            **kwargs
        )


@pytest.mark.parametrize(
    "threshold_name",
    [
        "pass_threshold",
        "fail_threshold"
    ]
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01
    ]
)
def test_out_of_range_thresholds_are_rejected(
    threshold_name,
    invalid_value
):
    pipeline, _, _ = build_pipeline(
        "Hypertension is associated with persistently "
        "elevated blood pressure [C1]."
    )

    kwargs = {
        "retriever": pipeline.retriever,
        "reranker": pipeline.reranker,
        "context_builder": pipeline.context_builder,
        "generator": pipeline.generator,
        "nli_provider": pipeline.nli_provider,
        "retrieval_k": 2,
        "rerank_k": 1
    }

    kwargs[threshold_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            f"{threshold_name} must be between 0 and 1"
        )
    ):
        SelfVerifyingRAG(
            **kwargs
        )


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        0.0,
        0.5,
        1,
        1.0
    ]
)
def test_valid_threshold_boundaries_are_accepted(
    threshold
):
    pipeline, _, _ = build_pipeline(
        "Hypertension is associated with persistently "
        "elevated blood pressure [C1]."
    )

    hardened_pipeline = SelfVerifyingRAG(
        retriever=pipeline.retriever,
        reranker=pipeline.reranker,
        context_builder=pipeline.context_builder,
        generator=pipeline.generator,
        nli_provider=pipeline.nli_provider,
        retrieval_k=2,
        rerank_k=1,
        pass_threshold=threshold,
        fail_threshold=threshold
    )

    assert hardened_pipeline.pass_threshold == float(
        threshold
    )
    assert hardened_pipeline.fail_threshold == float(
        threshold
    )

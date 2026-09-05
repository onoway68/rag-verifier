import pytest
from context_builder import ContextBuilder
from embedding_provider import FakeEmbeddingProvider
from generator import CitationAwareGenerator
from generator_provider import FakeGeneratorProvider
from nli_provider import FakeNLIProvider
from reranker import Reranker
from reranker_provider import FakeRerankerProvider
from retriever import Retriever
from self_verifying_rag import (
    OrchestratorStageError,
    SelfVerifyingRAG
)


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


class StubRetriever:
    def __init__(self, output):
        self.output = output

    def retrieve(self, question, top_k):
        return self.output


class StubReranker:
    def __init__(self, output):
        self.output = output

    def rerank(self, question, candidates, top_k):
        return self.output


class StubContextBuilder:
    def __init__(self, output):
        self.output = output

    def build(self, reranked):
        return self.output


class StubGenerator:
    def __init__(self, output):
        self.output = output

    def generate(
        self,
        question,
        context,
        citation_map
    ):
        return self.output


def test_retriever_output_must_be_list():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.retriever = StubRetriever(
        None
    )

    with pytest.raises(
        ValueError,
        match="retriever output must be a list"
    ):
        pipeline.run(question)


def test_reranker_output_must_be_list():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.reranker = StubReranker(
        None
    )

    with pytest.raises(
        ValueError,
        match="reranker output must be a list"
    ):
        pipeline.run(question)


@pytest.mark.parametrize(
    "invalid_output",
    [
        None,
        [],
        "invalid"
    ]
)
def test_context_builder_output_must_be_dict(
    invalid_output
):
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.context_builder = (
        StubContextBuilder(
            invalid_output
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "context_builder output "
            "must be a dict"
        )
    ):
        pipeline.run(question)


@pytest.mark.parametrize(
    "missing_key",
    [
        "context",
        "citation_map"
    ]
)
def test_context_builder_output_requires_keys(
    missing_key
):
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    output = {
        "context": "Evidence [C1]",
        "citation_map": {
            "C1": "Evidence"
        }
    }

    del output[missing_key]

    pipeline.context_builder = (
        StubContextBuilder(output)
    )

    with pytest.raises(
        ValueError,
        match=(
            f"context_builder output "
            f"must contain {missing_key}"
        )
    ):
        pipeline.run(question)


def test_context_builder_context_must_be_string():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.context_builder = (
        StubContextBuilder(
            {
                "context": None,
                "citation_map": {
                    "C1": "Evidence"
                }
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "context_builder context "
            "must be a string"
        )
    ):
        pipeline.run(question)


def test_context_builder_citation_map_must_be_dict():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.context_builder = (
        StubContextBuilder(
            {
                "context": "Evidence [C1]",
                "citation_map": None
            }
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "context_builder citation_map "
            "must be a dict"
        )
    ):
        pipeline.run(question)


@pytest.mark.parametrize(
    "invalid_output",
    [
        None,
        [],
        "invalid"
    ]
)
def test_generator_output_must_be_dict(
    invalid_output
):
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.generator = StubGenerator(
        invalid_output
    )

    with pytest.raises(
        ValueError,
        match="generator output must be a dict"
    ):
        pipeline.run(question)


def test_generator_output_requires_answer():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.generator = StubGenerator({})

    with pytest.raises(
        ValueError,
        match=(
            "generator output must contain answer"
        )
    ):
        pipeline.run(question)


def test_generator_answer_must_be_string():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.generator = StubGenerator(
        {
            "answer": None
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "generator answer must be a string"
        )
    ):
        pipeline.run(question)


class StubVerifier:
    def __init__(self, output):
        self.output = output

    def verify_answer(self, answer):
        return self.output


class StubVerifierFactory:
    def __init__(self, output):
        self.output = output

    def __call__(self, **kwargs):
        return StubVerifier(self.output)


def test_verifier_output_must_be_dict():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.verifier_factory = StubVerifierFactory(
        None
    )

    with pytest.raises(
        ValueError,
        match="verifier output must be a dict"
    ):
        pipeline.run(question)


@pytest.mark.parametrize(
    "missing_key",
    [
        "status",
        "trusted_release"
    ]
)
def test_verifier_output_requires_keys(
    missing_key
):
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    output = {
        "status": "PASS",
        "trusted_release": True
    }

    del output[missing_key]

    pipeline.verifier_factory = StubVerifierFactory(
        output
    )

    with pytest.raises(
        ValueError,
        match=(
            f"verifier output must contain "
            f"{missing_key}"
        )
    ):
        pipeline.run(question)


def test_verifier_status_must_be_valid():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.verifier_factory = StubVerifierFactory(
        {
            "status": "UNKNOWN",
            "trusted_release": False
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "verifier status must be "
            "PASS, REVIEW, or FAIL"
        )
    ):
        pipeline.run(question)


def test_verifier_trusted_release_must_be_bool():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.verifier_factory = StubVerifierFactory(
        {
            "status": "PASS",
            "trusted_release": 1
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "verifier trusted_release "
            "must be a bool"
        )
    ):
        pipeline.run(question)


@pytest.mark.parametrize(
    "status,trusted_release",
    [
        ("PASS", False),
        ("REVIEW", True),
        ("FAIL", True)
    ]
)
def test_verifier_status_and_trusted_release_must_be_consistent(
    status,
    trusted_release
):
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.verifier_factory = StubVerifierFactory(
        {
            "status": status,
            "trusted_release": trusted_release
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "verifier status and trusted_release "
            "are inconsistent"
        )
    ):
        pipeline.run(question)


def test_verifier_factory_must_not_be_none():
    pipeline, _, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    with pytest.raises(
        ValueError,
        match="verifier_factory must not be None"
    ):
        SelfVerifyingRAG(
            retriever=pipeline.retriever,
            reranker=pipeline.reranker,
            context_builder=pipeline.context_builder,
            generator=pipeline.generator,
            nli_provider=pipeline.nli_provider,
            retrieval_k=pipeline.retrieval_k,
            rerank_k=pipeline.rerank_k,
            pass_threshold=pipeline.pass_threshold,
            fail_threshold=pipeline.fail_threshold,
            verifier_factory=None
        )


def test_verifier_factory_must_be_callable():
    pipeline, _, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    with pytest.raises(
        ValueError,
        match="verifier_factory must be callable"
    ):
        SelfVerifyingRAG(
            retriever=pipeline.retriever,
            reranker=pipeline.reranker,
            context_builder=pipeline.context_builder,
            generator=pipeline.generator,
            nli_provider=pipeline.nli_provider,
            retrieval_k=pipeline.retrieval_k,
            rerank_k=pipeline.rerank_k,
            pass_threshold=pipeline.pass_threshold,
            fail_threshold=pipeline.fail_threshold,
            verifier_factory=42
        )


def test_verifier_factory_result_must_provide_verify_answer():
    pipeline, question, _ = build_pipeline(
        "Supported answer. [C1]"
    )

    pipeline.verifier_factory = (
        lambda **kwargs: object()
    )

    with pytest.raises(
        ValueError,
        match=(
            "verifier_factory must return an object "
            "with callable verify_answer"
        )
    ):
        pipeline.run(question)


class RecordingVerifierFactory:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return StubVerifier(self.output)


def test_verifier_factory_receives_exact_request_contract():
    pipeline, question, chunks = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    factory = RecordingVerifierFactory(
        {
            "status": "PASS",
            "trusted_release": True
        }
    )

    pipeline.verifier_factory = factory
    pipeline.pass_threshold = 0.91
    pipeline.fail_threshold = 0.87

    result = pipeline.run(question)

    assert len(factory.calls) == 1

    call = factory.calls[0]

    expected_citation_map = {
        "C1": chunks["hypertension-001"]
    }

    assert call["chunks"] == expected_citation_map
    assert list(call["retrieved_ids"]) == ["C1"]

    assert call["pass_threshold"] == 0.91
    assert call["fail_threshold"] == 0.87

    assert (
        call["nli_provider"]
        is pipeline.nli_provider
    )

    assert set(call) == {
        "chunks",
        "retrieved_ids",
        "pass_threshold",
        "fail_threshold",
        "nli_provider"
    }

    assert result["trusted_release"] is True


@pytest.mark.parametrize(
    "invalid_question",
    [
        None,
        123,
        "",
        "   "
    ]
)
def test_run_rejects_invalid_question(invalid_question):
    pipeline, _, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    with pytest.raises(
        ValueError,
        match="question must be a non-empty string"
    ):
        pipeline.run(invalid_question)


class RaisingRetriever:
    def retrieve(self, question, top_k):
        raise RuntimeError("retriever exploded")


class RaisingReranker:
    def rerank(self, question, candidates, top_k):
        raise RuntimeError("reranker exploded")


class RaisingContextBuilder:
    def build(self, reranked):
        raise RuntimeError("context builder exploded")


class RaisingGenerator:
    def generate(self, question, context, citation_map):
        raise RuntimeError("generator exploded")


class RaisingVerifier:
    def verify_answer(self, answer):
        raise RuntimeError("verifier exploded")


class RaisingVerifierFactory:
    def __call__(self, **kwargs):
        return RaisingVerifier()


def test_run_wraps_retriever_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.retriever = RaisingRetriever()

    with pytest.raises(
        OrchestratorStageError,
        match="retriever stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "retriever"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )
    assert str(
        exc_info.value.__cause__
    ) == "retriever exploded"


def test_run_wraps_reranker_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.reranker = RaisingReranker()

    with pytest.raises(
        OrchestratorStageError,
        match="reranker stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "reranker"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )


def test_run_wraps_context_builder_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.context_builder = RaisingContextBuilder()

    with pytest.raises(
        OrchestratorStageError,
        match="context_builder stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "context_builder"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )


def test_run_wraps_generator_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.generator = RaisingGenerator()

    with pytest.raises(
        OrchestratorStageError,
        match="generator stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "generator"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )


def test_run_wraps_verifier_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.verifier_factory = RaisingVerifierFactory()

    with pytest.raises(
        OrchestratorStageError,
        match="verifier stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "verifier"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )
    assert str(
        exc_info.value.__cause__
    ) == "verifier exploded"


class RaisingVerifierFactoryOnConstruction:
    def __call__(self, **kwargs):
        raise RuntimeError("verifier construction exploded")


def test_run_wraps_verifier_factory_exception():
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    pipeline.verifier_factory = (
        RaisingVerifierFactoryOnConstruction()
    )

    with pytest.raises(
        OrchestratorStageError,
        match="verifier stage failed"
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "verifier"
    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError
    )
    assert str(
        exc_info.value.__cause__
    ) == "verifier construction exploded"


class FakeReleasePolicy:
    def __init__(self, decision):
        self.decision = decision
        self.received = None

    def decide(self, verification):
        self.received = verification
        return self.decision


class RaisingReleasePolicy:
    def decide(self, verification):
        raise RuntimeError("policy unavailable")


def build_release_policy_pipeline(
    policy,
    answer=(
        "Hypertension is characterized by "
        "persistently elevated blood pressure. [C1]"
    )
):
    pipeline, question, chunks = build_pipeline(
        answer
    )

    pipeline.release_policy = policy

    return (
        pipeline,
        question,
        chunks
    )


def test_orchestrator_applies_release_policy():
    policy = FakeReleasePolicy({
        "action": "RELEASE",
        "releasable": True,
        "reason": "VERIFICATION_PASSED"
    })

    pipeline, question, _ = (
        build_release_policy_pipeline(
            policy
        )
    )

    result = pipeline.run(question)

    assert result["release_decision"] == {
        "action": "RELEASE",
        "releasable": True,
        "reason": "VERIFICATION_PASSED"
    }

    assert (
        policy.received
        is result["verification"]
    )


def test_orchestrator_uses_policy_releasability():
    policy = FakeReleasePolicy({
        "action": "HOLD_FOR_REVIEW",
        "releasable": False,
        "reason": "VERIFICATION_REQUIRES_REVIEW"
    })

    pipeline, question, _ = (
        build_release_policy_pipeline(
            policy
        )
    )

    result = pipeline.run(question)

    assert result["trusted_release"] is False


def test_release_policy_runtime_failure_is_stage_error():
    pipeline, question, _ = (
        build_release_policy_pipeline(
            RaisingReleasePolicy()
        )
    )

    with pytest.raises(
        OrchestratorStageError
    ) as exc_info:
        pipeline.run(question)

    assert exc_info.value.stage == "release_policy"


@pytest.mark.parametrize(
    "decision",
    [
        None,
        {},
        {
            "action": "RELEASE",
            "releasable": True
        },
        {
            "action": "UNKNOWN",
            "releasable": False,
            "reason": "INVALID"
        },
        {
            "action": "RELEASE",
            "releasable": "yes",
            "reason": "VERIFICATION_PASSED"
        },
    ]
)
def test_malformed_release_policy_output_is_rejected(
    decision
):
    policy = FakeReleasePolicy(
        decision
    )

    pipeline, question, _ = (
        build_release_policy_pipeline(
            policy
        )
    )

    with pytest.raises(ValueError):
        pipeline.run(question)


class StaticVerifier:
    def __init__(self, verification):
        self.verification = verification

    def verify_answer(self, answer):
        return dict(self.verification)


def test_invalid_release_policy_dependency_is_rejected():
    pipeline, _, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "release_policy must provide "
            r"callable decide\(\)"
        )
    ):
        SelfVerifyingRAG(
            retriever=pipeline.retriever,
            reranker=pipeline.reranker,
            context_builder=(
                pipeline.context_builder
            ),
            generator=pipeline.generator,
            nli_provider=pipeline.nli_provider,
            retrieval_k=pipeline.retrieval_k,
            rerank_k=pipeline.rerank_k,
            pass_threshold=(
                pipeline.pass_threshold
            ),
            fail_threshold=(
                pipeline.fail_threshold
            ),
            verifier_factory=(
                pipeline.verifier_factory
            ),
            release_policy=object()
        )


@pytest.mark.parametrize(
    (
        "status",
        "trusted_release",
        "expected_decision"
    ),
    [
        (
            "PASS",
            True,
            {
                "action": "RELEASE",
                "releasable": True,
                "reason": "VERIFICATION_PASSED"
            }
        ),
        (
            "REVIEW",
            False,
            {
                "action": "HOLD_FOR_REVIEW",
                "releasable": False,
                "reason": (
                    "VERIFICATION_REQUIRES_REVIEW"
                )
            }
        ),
        (
            "FAIL",
            False,
            {
                "action": "BLOCK",
                "releasable": False,
                "reason": "VERIFICATION_FAILED"
            }
        ),
    ]
)
def test_default_release_policy_maps_verification_status(
    status,
    trusted_release,
    expected_decision
):
    pipeline, question, _ = build_pipeline(
        (
            "Hypertension is characterized by "
            "persistently elevated blood pressure. [C1]"
        )
    )

    verification = {
        "status": status,
        "trusted_release": trusted_release
    }

    pipeline.verifier_factory = (
        lambda **kwargs: StaticVerifier(
            verification
        )
    )

    result = pipeline.run(question)

    assert (
        result["release_decision"]
        == expected_decision
    )

    assert (
        result["trusted_release"]
        is expected_decision["releasable"]
    )


@pytest.mark.parametrize(
    "decision,error_pattern",
    [
        (
            {
                "action": "RELEASE",
                "releasable": False,
                "reason": "VERIFICATION_PASSED"
            },
            (
                "release_policy action and "
                "releasable are inconsistent"
            )
        ),
        (
            {
                "action": "BLOCK",
                "releasable": False,
                "reason": "VERIFICATION_PASSED"
            },
            (
                "release_policy action and "
                "reason are inconsistent"
            )
        ),
    ]
)
def test_inconsistent_release_policy_decision_is_rejected(
    decision,
    error_pattern
):
    policy = FakeReleasePolicy(
        decision
    )

    pipeline, question, _ = (
        build_release_policy_pipeline(
            policy
        )
    )

    with pytest.raises(
        ValueError,
        match=error_pattern
    ):
        pipeline.run(question)

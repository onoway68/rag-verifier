import pytest

from generator import CitationAwareGenerator
from generator_provider import FakeGeneratorProvider


def make_generator(response):
    return CitationAwareGenerator(
        FakeGeneratorProvider(
            response=response
        )
    )


def test_generates_answer_with_valid_citation():
    generator = make_generator(
        "Diabetes can cause neuropathy [C1]."
    )

    result = generator.generate(
        question=(
            "What complication can diabetes cause?"
        ),
        context=(
            "[C1] Diabetes can cause neuropathy."
        ),
        citation_map={
            "C1": (
                "Diabetes can cause neuropathy."
            )
        }
    )

    assert result == {
        "answer": (
            "Diabetes can cause neuropathy [C1]."
        ),
        "citation_ids": ["C1"]
    }


def test_preserves_multiple_citations_in_order():
    generator = make_generator(
        (
            "Diabetes can cause neuropathy [C1]. "
            "Hypertension can cause cardiovascular "
            "complications [C2]."
        )
    )

    result = generator.generate(
        question="What complications are described?",
        context=(
            "[C1] Diabetes can cause neuropathy.\n\n"
            "[C2] Hypertension can cause "
            "cardiovascular complications."
        ),
        citation_map={
            "C1": (
                "Diabetes can cause neuropathy."
            ),
            "C2": (
                "Hypertension can cause "
                "cardiovascular complications."
            )
        }
    )

    assert result["citation_ids"] == [
        "C1",
        "C2"
    ]


def test_repeated_citations_are_preserved():
    generator = make_generator(
        (
            "Neuropathy is described [C1]. "
            "The same evidence also discusses "
            "diabetes [C1]."
        )
    )

    result = generator.generate(
        question="What does the evidence say?",
        context=(
            "[C1] Diabetes can cause neuropathy."
        ),
        citation_map={
            "C1": (
                "Diabetes can cause neuropathy."
            )
        }
    )

    assert result["citation_ids"] == [
        "C1",
        "C1"
    ]


def test_answer_without_citations_is_structurally_valid():
    generator = make_generator(
        "Diabetes can cause neuropathy."
    )

    result = generator.generate(
        question=(
            "What complication can diabetes cause?"
        ),
        context=(
            "[C1] Diabetes can cause neuropathy."
        ),
        citation_map={
            "C1": (
                "Diabetes can cause neuropathy."
            )
        }
    )

    assert result["citation_ids"] == []


def test_unknown_citation_is_rejected():
    generator = make_generator(
        "Diabetes can cause neuropathy [C99]."
    )

    with pytest.raises(
        ValueError,
        match="unknown citation IDs: C99"
    ):
        generator.generate(
            question=(
                "What complication can diabetes cause?"
            ),
            context=(
                "[C1] Diabetes can cause neuropathy."
            ),
            citation_map={
                "C1": (
                    "Diabetes can cause neuropathy."
                )
            }
        )


def test_multiple_unknown_citations_are_rejected():
    generator = make_generator(
        "Claim one [C99]. Claim two [C100]."
    )

    with pytest.raises(
        ValueError,
        match="C99, C100"
    ):
        generator.generate(
            question="What does the evidence say?",
            context="[C1] Supported evidence.",
            citation_map={
                "C1": "Supported evidence."
            }
        )


@pytest.mark.parametrize(
    "question",
    [
        None,
        "",
        "   ",
        123,
        [],
        {}
    ]
)
def test_invalid_question_is_rejected(question):
    generator = make_generator(
        "Answer [C1]."
    )

    with pytest.raises(
        ValueError,
        match=(
            "question must be a non-empty string"
        )
    ):
        generator.generate(
            question=question,
            context="[C1] Evidence.",
            citation_map={
                "C1": "Evidence."
            }
        )


@pytest.mark.parametrize(
    "context",
    [
        None,
        123,
        [],
        {}
    ]
)
def test_non_string_context_is_rejected(context):
    generator = make_generator(
        "Answer [C1]."
    )

    with pytest.raises(
        ValueError,
        match="context must be a string"
    ):
        generator.generate(
            question="Question?",
            context=context,
            citation_map={
                "C1": "Evidence."
            }
        )


@pytest.mark.parametrize(
    "citation_map",
    [
        None,
        [],
        "",
        123
    ]
)
def test_non_dictionary_citation_map_is_rejected(
    citation_map
):
    generator = make_generator(
        "Answer."
    )

    with pytest.raises(
        ValueError,
        match=(
            "citation_map must be a dictionary"
        )
    ):
        generator.generate(
            question="Question?",
            context="Evidence.",
            citation_map=citation_map
        )


@pytest.mark.parametrize(
    "citation_id",
    [
        None,
        "",
        "   ",
        123
    ]
)
def test_invalid_citation_id_is_rejected(
    citation_id
):
    generator = make_generator(
        "Answer."
    )

    with pytest.raises(
        ValueError,
        match=(
            "citation IDs must be "
            "non-empty strings"
        )
    ):
        generator.generate(
            question="Question?",
            context="Evidence.",
            citation_map={
                citation_id: "Evidence."
            }
        )


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        "",
        "   ",
        123,
        [],
        {}
    ]
)
def test_invalid_citation_evidence_is_rejected(
    evidence
):
    generator = make_generator(
        "Answer."
    )

    with pytest.raises(
        ValueError,
        match=(
            "citation evidence must be "
            "a non-empty string"
        )
    ):
        generator.generate(
            question="Question?",
            context="Evidence.",
            citation_map={
                "C1": evidence
            }
        )


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "   ",
        123,
        [],
        {}
    ]
)
def test_invalid_provider_response_is_rejected(
    response
):
    generator = make_generator(
        response
    )

    with pytest.raises(
        ValueError,
        match=(
            "provider must return "
            "a non-empty string"
        )
    ):
        generator.generate(
            question="Question?",
            context="[C1] Evidence.",
            citation_map={
                "C1": "Evidence."
            }
        )


def test_missing_provider_is_rejected():
    with pytest.raises(
        ValueError,
        match="provider is required"
    ):
        CitationAwareGenerator(None)


def test_provider_without_generate_is_rejected():
    class InvalidProvider:
        pass

    with pytest.raises(
        ValueError,
        match=(
            "provider must define generate"
        )
    ):
        CitationAwareGenerator(
            InvalidProvider()
        )


def test_provider_generate_must_be_callable():
    class InvalidProvider:
        generate = "not callable"

    with pytest.raises(
        ValueError,
        match=(
            "provider must define generate"
        )
    ):
        CitationAwareGenerator(
            InvalidProvider()
        )


class RecordingGeneratorProvider:
    def __init__(self, response="Answer [C1]."):
        self.response = response
        self.calls = []

    def generate(
        self,
        question,
        context
    ):
        self.calls.append(
            {
                "question": question,
                "context": context
            }
        )
        return self.response


def test_valid_request_calls_provider_once():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    result = generator.generate(
        question="What does the evidence say?",
        context="[C1] Evidence.",
        citation_map={
            "C1": "Evidence."
        }
    )

    assert result["answer"] == "Answer [C1]."

    assert provider.calls == [
        {
            "question": (
                "What does the evidence say?"
            ),
            "context": "[C1] Evidence."
        }
    ]


def test_invalid_question_fails_before_provider_call():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(ValueError):
        generator.generate(
            question="",
            context="[C1] Evidence.",
            citation_map={
                "C1": "Evidence."
            }
        )

    assert provider.calls == []


def test_invalid_context_fails_before_provider_call():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(ValueError):
        generator.generate(
            question="Question?",
            context=None,
            citation_map={
                "C1": "Evidence."
            }
        )

    assert provider.calls == []


def test_invalid_citation_map_fails_before_provider_call():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(ValueError):
        generator.generate(
            question="Question?",
            context="[C1] Evidence.",
            citation_map=None
        )

    assert provider.calls == []


def test_context_citation_missing_from_map_is_rejected():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(
        ValueError,
        match=(
            "context citation IDs must match "
            "citation_map"
        )
    ):
        generator.generate(
            question="Question?",
            context="[C1] Evidence.",
            citation_map={
                "C2": "Evidence."
            }
        )

    assert provider.calls == []


def test_citation_map_entry_missing_from_context_is_rejected():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(
        ValueError,
        match=(
            "context citation IDs must match "
            "citation_map"
        )
    ):
        generator.generate(
            question="Question?",
            context="[C1] Evidence.",
            citation_map={
                "C1": "Evidence.",
                "C2": "Other evidence."
            }
        )

    assert provider.calls == []


def test_empty_evidence_fails_before_provider_call():
    provider = RecordingGeneratorProvider()

    generator = CitationAwareGenerator(
        provider
    )

    with pytest.raises(
        ValueError,
        match="grounding evidence is required"
    ):
        generator.generate(
            question="What does the evidence say?",
            context="",
            citation_map={}
        )

    assert provider.calls == []

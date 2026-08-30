import pytest

pytestmark = pytest.mark.integration
from verifier import RAGVerifier


@pytest.fixture(scope="module")
def verifier():

    chunks = {
        "mi-001": (
            "Myocardial infarction occurs when blood flow to part "
            "of the heart muscle is blocked or severely reduced."
        ),

        "mi-002": (
            "Symptoms may include chest discomfort, shortness of "
            "breath, nausea, or sweating."
        ),

        "htn-001": (
            "Hypertension is persistent elevation of arterial blood pressure."
        ),

        "pna-001": (
            "Pneumonia is an infection involving the lung parenchyma."
        )
    }

    retrieved_ids = {
        "mi-001",
        "mi-002",
        "htn-001"
    }

    return RAGVerifier(
        chunks=chunks,
        retrieved_ids=retrieved_ids
    )


def test_supported_citation(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. [mi-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "PASS"
    assert result["trusted_release"] is True
    assert result["citation_coverage"]["coverage_percent"] == 100.0


def test_fabricated_citation_fails(verifier):

    answer = (
        "A heart attack occurs when blood flow is blocked. "
        "[fake-999]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False

    verification = result["claims"][0]["verification"][0]

    assert verification["reason"] == "CITATION_NOT_FOUND"


def test_existing_but_unretrieved_citation_fails(verifier):

    answer = (
        "Pneumonia is an infection involving the lung parenchyma. "
        "[pna-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"

    verification = result["claims"][0]["verification"][0]

    assert verification["reason"] == "CITATION_NOT_RETRIEVED"


def test_uncited_factual_claim_fails(verifier):

    answer = (
        "A heart attack occurs when blood flow is blocked."
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False
    assert result["citation_coverage"]["coverage_percent"] == 0.0
    assert result["claims"][0]["reason"] == "UNCITED_FACTUAL_CLAIM"


def test_unsupported_citation_requires_review(verifier):

    answer = (
        "The heart attack occurred on a Monday. [mi-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "REVIEW"
    assert result["trusted_release"] is False

    verification = result["claims"][0]["verification"][0]

    assert verification["reason"] == "NLI_CONFIDENCE_BELOW_DECISION_THRESHOLD"


def test_contradictory_citation_fails(verifier):

    answer = (
        "A heart attack is caused by pneumonia. [mi-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"

    verification = result["claims"][0]["verification"][0]

    assert verification["reason"] == "CONTRADICTION_THRESHOLD_MET"


def test_valid_plus_fabricated_citation_fails(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. "
        "[mi-001] [fake-999]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"

    statuses = [
        item["status"]
        for item in result["claims"][0]["verification"]
    ]

    assert statuses == ["PASS", "FAIL"]


def test_valid_symptom_citation_passes(verifier):

    answer = (
        "Symptoms may include chest discomfort and sweating. "
        "[mi-002]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "PASS"


def test_two_supported_claims_pass(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. [mi-001] "
        "Symptoms may include chest discomfort and sweating. [mi-002]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "PASS"
    assert result["trusted_release"] is True
    assert len(result["claims"]) == 2
    assert result["citation_coverage"]["coverage_percent"] == 100.0


def test_supported_plus_uncited_claim_fails(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. [mi-001] "
        "Immediate surgery is always required."
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False
    assert result["citation_coverage"]["coverage_percent"] == 50.0

    assert result["claims"][0]["status"] == "PASS"
    assert result["claims"][1]["status"] == "FAIL"
    assert result["claims"][1]["reason"] == "UNCITED_FACTUAL_CLAIM"


def test_supported_plus_neutral_claim_requires_review(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. [mi-001] "
        "The heart attack occurred on a Monday. [mi-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "REVIEW"
    assert result["trusted_release"] is False

    assert result["claims"][0]["status"] == "PASS"
    assert result["claims"][1]["status"] == "REVIEW"


def test_supported_plus_fabricated_claim_fails(verifier):

    answer = (
        "Symptoms may include chest discomfort and sweating. [mi-002] "
        "Hypertension is persistent elevation of arterial blood pressure. "
        "[fake-999]"
    )

    result = verifier.verify_answer(answer)

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False

    assert result["claims"][0]["status"] == "PASS"
    assert result["claims"][1]["status"] == "FAIL"


def test_empty_answer_fails_closed(verifier):

    result = verifier.verify_answer("")

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False
    assert result["reason"] == "NO_VERIFIABLE_CLAIMS"


def test_all_claims_must_pass_for_trusted_release(verifier):

    answer = (
        "A heart attack occurs when blood flow to part of the "
        "heart muscle is blocked or severely reduced. [mi-001] "
        "The heart attack occurred on a Monday. [mi-001]"
    )

    result = verifier.verify_answer(answer)

    assert result["trusted_release"] is False


def test_atomic_supported_conjunction_passes(verifier):

    verifier.chunks["combo-001"] = (
        "The patient has hypertension. "
        "The patient has pneumonia."
    )

    verifier.retrieved_ids.add(
        "combo-001"
    )

    answer = (
        "The patient has hypertension and "
        "the patient has pneumonia. [combo-001]"
    )

    result = verifier.verify_answer(
        answer
    )

    assert len(result["claims"]) == 2
    assert result["claims"][0]["status"] == "PASS"
    assert result["claims"][1]["status"] == "PASS"
    assert result["status"] == "PASS"
    assert result["trusted_release"] is True


def test_atomic_partial_support_does_not_fully_pass(verifier):

    verifier.chunks["htn-only-001"] = (
        "The patient has hypertension."
    )

    verifier.retrieved_ids.add(
        "htn-only-001"
    )

    answer = (
        "The patient has hypertension and "
        "the patient has pneumonia. [htn-only-001]"
    )

    result = verifier.verify_answer(
        answer
    )

    assert len(result["claims"]) == 2
    assert result["claims"][0]["status"] == "PASS"

    assert result["claims"][1]["status"] in {
        "REVIEW",
        "FAIL"
    }

    assert result["status"] in {
        "REVIEW",
        "FAIL"
    }

    assert result["trusted_release"] is False


def test_non_atomic_and_phrase_is_not_split(verifier):

    answer = (
        "Symptoms may include chest discomfort and sweating. "
        "[mi-002]"
    )

    result = verifier.verify_answer(
        answer
    )

    assert len(result["claims"]) == 1
    assert result["status"] == "PASS"


def test_atomic_uncited_sentence_fails(verifier):

    answer = (
        "The patient has hypertension and "
        "the patient has pneumonia."
    )

    result = verifier.verify_answer(
        answer
    )

    assert len(result["claims"]) == 2

    assert all(
        claim["status"] == "FAIL"
        for claim in result["claims"]
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_threshold_entailment_above_pass_threshold_passes(verifier):

    scores = {
        "entailment": 0.95,
        "contradiction": 0.02,
        "neutral": 0.03
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "PASS"
    assert result["reason"] == "ENTAILMENT_THRESHOLD_MET"


def test_threshold_entailment_exact_boundary_passes(verifier):

    scores = {
        "entailment": verifier.pass_threshold,
        "contradiction": 0.01,
        "neutral": 1.0 - verifier.pass_threshold - 0.01
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "PASS"


def test_argmax_entailment_below_threshold_requires_review(verifier):

    scores = {
        "entailment": 0.58,
        "contradiction": 0.02,
        "neutral": 0.40
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "REVIEW"
    assert result["decision_label"] == "entailment"


def test_threshold_contradiction_above_fail_threshold_fails(verifier):

    scores = {
        "entailment": 0.01,
        "contradiction": 0.97,
        "neutral": 0.02
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "FAIL"
    assert result["reason"] == "CONTRADICTION_THRESHOLD_MET"


def test_threshold_contradiction_exact_boundary_fails(verifier):

    scores = {
        "entailment": 0.01,
        "contradiction": verifier.fail_threshold,
        "neutral": 1.0 - verifier.fail_threshold - 0.01
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "FAIL"


def test_argmax_contradiction_below_fail_threshold_requires_review(verifier):

    scores = {
        "entailment": 0.10,
        "contradiction": 0.60,
        "neutral": 0.30
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "REVIEW"
    assert result["decision_label"] == "contradiction"


def test_neutral_dominant_requires_review(verifier):

    scores = {
        "entailment": 0.10,
        "contradiction": 0.10,
        "neutral": 0.80
    }

    result = verifier.classify_nli_scores(
        scores
    )

    assert result["status"] == "REVIEW"
    assert result["decision_label"] == "neutral"


def test_invalid_pass_threshold_rejected():

    with pytest.raises(ValueError):

        RAGVerifier(
            chunks={},
            retrieved_ids=set(),
            pass_threshold=1.1
        )


def test_invalid_fail_threshold_rejected():

    with pytest.raises(ValueError):

        RAGVerifier(
            chunks={},
            retrieved_ids=set(),
            fail_threshold=-0.1
        )



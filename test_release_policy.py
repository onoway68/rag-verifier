import pytest

from release_policy import ReleasePolicy


def verification(status, trusted_release):
    return {
        "status": status,
        "trusted_release": trusted_release
    }


def test_pass_releases_answer():
    policy = ReleasePolicy()

    result = policy.decide(
        verification("PASS", True)
    )

    assert result == {
        "action": "RELEASE",
        "releasable": True,
        "reason": "VERIFICATION_PASSED"
    }


def test_review_holds_for_human_review():
    policy = ReleasePolicy()

    result = policy.decide(
        verification("REVIEW", False)
    )

    assert result == {
        "action": "HOLD_FOR_REVIEW",
        "releasable": False,
        "reason": "VERIFICATION_REQUIRES_REVIEW"
    }


def test_fail_blocks_answer():
    policy = ReleasePolicy()

    result = policy.decide(
        verification("FAIL", False)
    )

    assert result == {
        "action": "BLOCK",
        "releasable": False,
        "reason": "VERIFICATION_FAILED"
    }


@pytest.mark.parametrize(
    "status,trusted_release",
    [
        ("PASS", False),
        ("REVIEW", True),
        ("FAIL", True),
    ]
)
def test_inconsistent_verification_contract_is_rejected(
    status,
    trusted_release
):
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match=(
            "verification status and "
            "trusted_release are inconsistent"
        )
    ):
        policy.decide(
            verification(
                status,
                trusted_release
            )
        )


def test_non_dict_verification_is_rejected():
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match="verification must be a dict"
    ):
        policy.decide("PASS")


def test_missing_status_is_rejected():
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match="verification must contain status"
    ):
        policy.decide({
            "trusted_release": False
        })


def test_missing_trusted_release_is_rejected():
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match=(
            "verification must contain "
            "trusted_release"
        )
    ):
        policy.decide({
            "status": "PASS"
        })


def test_invalid_status_is_rejected():
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match=(
            "verification status must be "
            "PASS, REVIEW, or FAIL"
        )
    ):
        policy.decide(
            verification(
                "UNKNOWN",
                False
            )
        )


@pytest.mark.parametrize(
    "trusted_release",
    [
        1,
        0,
        "true",
        None,
    ]
)
def test_non_boolean_trusted_release_is_rejected(
    trusted_release
):
    policy = ReleasePolicy()

    with pytest.raises(
        ValueError,
        match="trusted_release must be a bool"
    ):
        policy.decide({
            "status": "PASS",
            "trusted_release": trusted_release
        })

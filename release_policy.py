class ReleasePolicy:
    def decide(self, verification):
        if not isinstance(verification, dict):
            raise ValueError(
                "verification must be a dict"
            )

        if "status" not in verification:
            raise ValueError(
                "verification must contain status"
            )

        if "trusted_release" not in verification:
            raise ValueError(
                "verification must contain trusted_release"
            )

        status = verification["status"]
        trusted_release = verification[
            "trusted_release"
        ]

        if status not in (
            "PASS",
            "REVIEW",
            "FAIL"
        ):
            raise ValueError(
                "verification status must be "
                "PASS, REVIEW, or FAIL"
            )

        if not isinstance(
            trusted_release,
            bool
        ):
            raise ValueError(
                "trusted_release must be a bool"
            )

        expected_trusted_release = (
            status == "PASS"
        )

        if (
            trusted_release
            != expected_trusted_release
        ):
            raise ValueError(
                "verification status and "
                "trusted_release are inconsistent"
            )

        if status == "PASS":
            return {
                "action": "RELEASE",
                "releasable": True,
                "reason": "VERIFICATION_PASSED"
            }

        if status == "REVIEW":
            return {
                "action": "HOLD_FOR_REVIEW",
                "releasable": False,
                "reason": (
                    "VERIFICATION_REQUIRES_REVIEW"
                )
            }

        return {
            "action": "BLOCK",
            "releasable": False,
            "reason": "VERIFICATION_FAILED"
        }

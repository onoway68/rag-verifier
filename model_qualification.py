import math


class ModelQualificationPolicy:
    def __init__(
        self,
        pass_thresholds,
        review_thresholds
    ):
        if (
            not isinstance(pass_thresholds, dict)
            or not isinstance(review_thresholds, dict)
            or not pass_thresholds
            or not review_thresholds
        ):
            raise ValueError(
                "thresholds must be non-empty dicts"
            )

        if set(pass_thresholds) != set(review_thresholds):
            raise ValueError(
                "pass and review thresholds must "
                "use the same metrics"
            )

        for metric in pass_thresholds:
            pass_threshold = pass_thresholds[metric]
            review_threshold = review_thresholds[metric]

            for threshold in (
                pass_threshold,
                review_threshold
            ):
                if (
                    isinstance(threshold, bool)
                    or not isinstance(
                        threshold,
                        (int, float)
                    )
                    or not math.isfinite(threshold)
                    or threshold < 0.0
                    or threshold > 1.0
                ):
                    raise ValueError(
                        "thresholds must be numeric "
                        "values between 0 and 1"
                    )

            if pass_threshold < review_threshold:
                raise ValueError(
                    "pass threshold must be greater than "
                    "or equal to review threshold"
                )

        self.pass_thresholds = dict(pass_thresholds)
        self.review_thresholds = dict(review_thresholds)

    def decide(self, metrics):
        if not isinstance(metrics, dict):
            raise ValueError(
                "qualification metrics must be a dict"
            )

        required_metrics = set(self.pass_thresholds)
        missing = required_metrics - set(metrics)

        if missing:
            raise ValueError(
                "qualification metrics are missing"
            )

        for metric in required_metrics:
            value = metrics[metric]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    "qualification metrics must be "
                    "numeric values between 0 and 1"
                )

        passes = all(
            metrics[metric] >= self.pass_thresholds[metric]
            for metric in required_metrics
        )

        if passes:
            return {
                "status": "PASS",
                "reason": "QUALIFICATION_THRESHOLDS_SATISFIED"
            }

        reviews = all(
            metrics[metric] >= self.review_thresholds[metric]
            for metric in required_metrics
        )

        if reviews:
            return {
                "status": "REVIEW",
                "reason": "PASS_THRESHOLDS_NOT_SATISFIED"
            }

        return {
            "status": "FAIL",
            "reason": "REVIEW_THRESHOLDS_NOT_SATISFIED"
        }


class ModelQualificationRecordBuilder:
    def __init__(self, policy):
        decide = getattr(policy, "decide", None)

        if not callable(decide):
            raise ValueError(
                "policy must provide callable decide()"
            )

        self.policy = policy

    @staticmethod
    def _validate_non_empty_string(value, field_name):
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

    @staticmethod
    def _validate_positive_integer(value, field_name):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a positive integer"
            )

    @staticmethod
    def _validate_metric_value(value, field_name):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(
                f"{field_name} must be a finite numeric "
                "value between 0 and 1"
            )

    @staticmethod
    def _validate_qualification_output(output):
        if not isinstance(output, dict):
            raise ValueError(
                "qualification output is invalid"
            )

        required = {"status", "reason"}

        if not required.issubset(output):
            raise ValueError(
                "qualification output is invalid"
            )

        status = output["status"]
        reason = output["reason"]

        if (
            not isinstance(status, str)
            or status not in {"PASS", "REVIEW", "FAIL"}
        ):
            raise ValueError(
                "qualification output is invalid"
            )

        if (
            not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(
                "qualification output is invalid"
            )

    def build(self, model, benchmark, metrics):
        if not isinstance(model, dict):
            raise ValueError("model must be a dict")

        if not isinstance(benchmark, dict):
            raise ValueError("benchmark must be a dict")

        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dict")

        required_model = {
            "model_id",
            "model_revision",
            "provider_type",
            "embedding_dimension"
        }

        if not required_model.issubset(model):
            raise ValueError(
                "model metadata is incomplete"
            )

        required_benchmark = {
            "benchmark_id",
            "benchmark_version",
            "top_k"
        }

        if not required_benchmark.issubset(benchmark):
            raise ValueError(
                "benchmark metadata is incomplete"
            )

        required_metrics = {
            "query_count",
            "recall_at_k",
            "mrr",
            "precision_at_k"
        }

        if not required_metrics.issubset(metrics):
            raise ValueError("metrics are incomplete")

        self._validate_non_empty_string(
            model["model_id"],
            "model_id"
        )
        self._validate_non_empty_string(
            model["model_revision"],
            "model_revision"
        )
        self._validate_non_empty_string(
            model["provider_type"],
            "provider_type"
        )
        self._validate_positive_integer(
            model["embedding_dimension"],
            "embedding_dimension"
        )
        self._validate_non_empty_string(
            benchmark["benchmark_id"],
            "benchmark_id"
        )
        self._validate_non_empty_string(
            benchmark["benchmark_version"],
            "benchmark_version"
        )
        self._validate_positive_integer(
            benchmark["top_k"],
            "top_k"
        )
        self._validate_positive_integer(
            metrics["query_count"],
            "query_count"
        )
        self._validate_metric_value(
            metrics["recall_at_k"],
            "recall_at_k"
        )
        self._validate_metric_value(
            metrics["mrr"],
            "mrr"
        )
        self._validate_metric_value(
            metrics["precision_at_k"],
            "precision_at_k"
        )

        qualification_metrics = {
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "precision_at_k": metrics["precision_at_k"]
        }

        qualification = self.policy.decide(
            qualification_metrics
        )

        self._validate_qualification_output(qualification)

        qualification = {
            "status": qualification["status"],
            "reason": qualification["reason"]
        }

        return {
            "model": {
                "model_id": model["model_id"],
                "model_revision": model["model_revision"],
                "provider_type": model["provider_type"],
                "embedding_dimension": model[
                    "embedding_dimension"
                ]
            },
            "benchmark": {
                "benchmark_id": benchmark["benchmark_id"],
                "benchmark_version": benchmark[
                    "benchmark_version"
                ],
                "top_k": benchmark["top_k"],
                "query_count": metrics["query_count"]
            },
            "metrics": {
                "recall_at_k": metrics["recall_at_k"],
                "mrr": metrics["mrr"],
                "precision_at_k": metrics["precision_at_k"]
            },
            "qualification": qualification
        }

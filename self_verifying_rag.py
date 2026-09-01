from verifier import RAGVerifier


class OrchestratorStageError(RuntimeError):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(
            f"{stage} stage failed"
        )


class SelfVerifyingRAG:
    def __init__(
        self,
        retriever,
        reranker,
        context_builder,
        generator,
        nli_provider,
        retrieval_k=5,
        rerank_k=3,
        pass_threshold=0.90,
        fail_threshold=0.90,
        verifier_factory=RAGVerifier
    ):
        for name, value in (
            ("retrieval_k", retrieval_k),
            ("rerank_k", rerank_k)
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )

        if rerank_k > retrieval_k:
            raise ValueError(
                "rerank_k must not exceed retrieval_k"
            )

        dependencies = (
            ("retriever", retriever, "retrieve"),
            ("reranker", reranker, "rerank"),
            (
                "context_builder",
                context_builder,
                "build"
            ),
            ("generator", generator, "generate"),
            (
                "nli_provider",
                nli_provider,
                "predict"
            )
        )

        for name, dependency, method_name in dependencies:
            if dependency is None:
                raise ValueError(
                    f"{name} must not be None"
                )

            method = getattr(
                dependency,
                method_name,
                None
            )

            if not callable(method):
                raise ValueError(
                    f"{name} must provide callable "
                    f"{method_name}()"
                )

        for name, value in (
            ("pass_threshold", pass_threshold),
            ("fail_threshold", fail_threshold)
        ):
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float)
                )
            ):
                raise ValueError(
                    f"{name} must be numeric"
                )

            value = float(value)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        pass_threshold = float(
            pass_threshold
        )
        fail_threshold = float(
            fail_threshold
        )

        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.nli_provider = nli_provider

        if verifier_factory is None:
            raise ValueError(
                "verifier_factory must not be None"
            )

        if not callable(verifier_factory):
            raise ValueError(
                "verifier_factory must be callable"
            )

        self.verifier_factory = verifier_factory

        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

    @staticmethod
    def _validate_retriever_output(output):
        if not isinstance(output, list):
            raise ValueError(
                "retriever output must be a list"
            )

    @staticmethod
    def _validate_reranker_output(output):
        if not isinstance(output, list):
            raise ValueError(
                "reranker output must be a list"
            )

    @staticmethod
    def _validate_context_builder_output(output):
        if not isinstance(output, dict):
            raise ValueError(
                "context_builder output must be a dict"
            )

        for key in (
            "context",
            "citation_map"
        ):
            if key not in output:
                raise ValueError(
                    "context_builder output "
                    f"must contain {key}"
                )

        if not isinstance(
            output["context"],
            str
        ):
            raise ValueError(
                "context_builder context "
                "must be a string"
            )

        if not isinstance(
            output["citation_map"],
            dict
        ):
            raise ValueError(
                "context_builder citation_map "
                "must be a dict"
            )

    @staticmethod
    def _validate_generator_output(output):
        if not isinstance(output, dict):
            raise ValueError(
                "generator output must be a dict"
            )

        if "answer" not in output:
            raise ValueError(
                "generator output must contain answer"
            )

        if not isinstance(
            output["answer"],
            str
        ):
            raise ValueError(
                "generator answer must be a string"
            )

    @staticmethod
    def _validate_verifier_output(output):
        if not isinstance(output, dict):
            raise ValueError(
                "verifier output must be a dict"
            )

        for key in (
            "status",
            "trusted_release"
        ):
            if key not in output:
                raise ValueError(
                    "verifier output "
                    f"must contain {key}"
                )

        status = output["status"]
        trusted_release = output[
            "trusted_release"
        ]

        if status not in (
            "PASS",
            "REVIEW",
            "FAIL"
        ):
            raise ValueError(
                "verifier status must be "
                "PASS, REVIEW, or FAIL"
            )

        if not isinstance(
            trusted_release,
            bool
        ):
            raise ValueError(
                "verifier trusted_release "
                "must be a bool"
            )

        expected_release = (
            status == "PASS"
        )

        if trusted_release != expected_release:
            raise ValueError(
                "verifier status and trusted_release "
                "are inconsistent"
            )

    def run(
        self,
        question
    ):
        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "question must be a non-empty string"
            )

        try:
            candidates = self.retriever.retrieve(
                question,
                top_k=self.retrieval_k
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "retriever"
            ) from exc

        self._validate_retriever_output(
            candidates
        )

        try:
            reranked = self.reranker.rerank(
                question,
                candidates,
                top_k=self.rerank_k
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "reranker"
            ) from exc

        self._validate_reranker_output(
            reranked
        )

        try:
            context_result = (
                self.context_builder.build(
                    reranked
                )
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "context_builder"
            ) from exc

        self._validate_context_builder_output(
            context_result
        )

        try:
            generation_result = (
                self.generator.generate(
                    question=question,
                    context=context_result["context"],
                    citation_map=(
                        context_result[
                            "citation_map"
                        ]
                    )
                )
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "generator"
            ) from exc

        self._validate_generator_output(
            generation_result
        )

        citation_map = (
            context_result[
                "citation_map"
            ]
        )

        try:
            verifier = self.verifier_factory(
                chunks=citation_map,
                retrieved_ids=(
                    citation_map.keys()
                ),
                pass_threshold=(
                    self.pass_threshold
                ),
                fail_threshold=(
                    self.fail_threshold
                ),
                nli_provider=(
                    self.nli_provider
                )
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "verifier"
            ) from exc

        verify_answer = getattr(
            verifier,
            "verify_answer",
            None
        )

        if not callable(verify_answer):
            raise ValueError(
                "verifier_factory must return an object "
                "with callable verify_answer"
            )

        try:
            verification = (
                verify_answer(
                    generation_result[
                        "answer"
                    ]
                )
            )
        except Exception as exc:
            raise OrchestratorStageError(
                "verifier"
            ) from exc

        self._validate_verifier_output(
            verification
        )

        return {
            "question": question,
            "retrieval": candidates,
            "reranked_evidence": reranked,
            "context": context_result,
            "generation": generation_result,
            "verification": verification,
            "trusted_release": (
                verification[
                    "trusted_release"
                ]
            )
        }

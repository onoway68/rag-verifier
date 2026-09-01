from verifier import RAGVerifier


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
        fail_threshold=0.90
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

    def run(
        self,
        question
    ):
        candidates = self.retriever.retrieve(
            question,
            top_k=self.retrieval_k
        )

        self._validate_retriever_output(
            candidates
        )

        reranked = self.reranker.rerank(
            question,
            candidates,
            top_k=self.rerank_k
        )

        self._validate_reranker_output(
            reranked
        )

        context_result = (
            self.context_builder.build(
                reranked
            )
        )

        self._validate_context_builder_output(
            context_result
        )

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

        self._validate_generator_output(
            generation_result
        )

        citation_map = (
            context_result[
                "citation_map"
            ]
        )

        verifier = RAGVerifier(
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

        verification = (
            verifier.verify_answer(
                generation_result[
                    "answer"
                ]
            )
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

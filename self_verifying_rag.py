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

        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.nli_provider = nli_provider

        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

    def run(
        self,
        question
    ):
        candidates = self.retriever.retrieve(
            question,
            top_k=self.retrieval_k
        )

        reranked = self.reranker.rerank(
            question,
            candidates,
            top_k=self.rerank_k
        )

        context_result = (
            self.context_builder.build(
                reranked
            )
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

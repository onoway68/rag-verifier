from retrieval_evaluator import (
    RetrievalEvaluator
)


def evaluate_reranking(
    retriever,
    reranker,
    cases,
    candidate_k=5,
    final_k=3
):
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or candidate_k <= 0
    ):
        raise ValueError(
            "candidate_k must be a positive integer"
        )

    if (
        isinstance(final_k, bool)
        or not isinstance(final_k, int)
        or final_k <= 0
    ):
        raise ValueError(
            "final_k must be a positive integer"
        )

    if final_k > candidate_k:
        raise ValueError(
            "final_k must not exceed candidate_k"
        )

    evaluator = RetrievalEvaluator()

    baseline_cases = []
    reranked_cases = []
    evaluated_cases = []

    for case in cases:
        query = case["query"]

        relevant_ids = list(
            case["relevant_ids"]
        )

        candidates = retriever.retrieve(
            query=query,
            top_k=candidate_k
        )

        baseline_ids = [
            candidate["chunk_id"]
            for candidate in candidates[
                :final_k
            ]
        ]

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=final_k
        )

        reranked_ids = [
            candidate["chunk_id"]
            for candidate in reranked
        ]

        baseline_metrics = (
            evaluator.evaluate_query(
                retrieved_ids=baseline_ids,
                relevant_ids=relevant_ids,
                k=final_k
            )
        )

        reranked_metrics = (
            evaluator.evaluate_query(
                retrieved_ids=reranked_ids,
                relevant_ids=relevant_ids,
                k=final_k
            )
        )

        baseline_cases.append(
            {
                "retrieved_ids": baseline_ids,
                "relevant_ids": relevant_ids
            }
        )

        reranked_cases.append(
            {
                "retrieved_ids": reranked_ids,
                "relevant_ids": relevant_ids
            }
        )

        evaluated_cases.append(
            {
                "query": query,
                "relevant_ids": relevant_ids,
                "candidate_ids": [
                    candidate["chunk_id"]
                    for candidate in candidates
                ],
                "baseline_ids": baseline_ids,
                "reranked_ids": reranked_ids,
                "baseline_metrics": (
                    baseline_metrics
                ),
                "reranked_metrics": (
                    reranked_metrics
                )
            }
        )

    baseline_summary = (
        evaluator.evaluate_dataset(
            cases=baseline_cases,
            k=final_k
        )
    )

    reranked_summary = (
        evaluator.evaluate_dataset(
            cases=reranked_cases,
            k=final_k
        )
    )

    return {
        "baseline": baseline_summary,
        "reranked": reranked_summary,
        "cases": evaluated_cases
    }

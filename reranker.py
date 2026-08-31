import math


class Reranker:
    def __init__(
        self,
        reranker_provider
    ):
        self.reranker_provider = (
            reranker_provider
        )

    def validate_scores(
        self,
        scores,
        expected_count
    ):
        if not isinstance(scores, list):
            raise ValueError(
                "Reranker provider output must be a list"
            )

        if len(scores) != expected_count:
            raise ValueError(
                "Reranker provider returned an unexpected number of scores"
            )

        for score in scores:
            if (
                isinstance(score, bool)
                or not isinstance(
                    score,
                    (int, float)
                )
            ):
                raise ValueError(
                    "Reranker scores must be numeric"
                )

            if not math.isfinite(score):
                raise ValueError(
                    "Reranker scores must be finite"
                )

        return scores

    def rerank(
        self,
        query,
        candidates,
        top_k=None
    ):
        if not isinstance(candidates, list):
            raise ValueError(
                "candidates must be a list"
            )

        if (
            top_k is not None
            and (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or top_k <= 0
            )
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        if not candidates:
            return []

        texts = []

        for candidate in candidates:
            if not isinstance(
                candidate,
                dict
            ):
                raise ValueError(
                    "Each candidate must be a dictionary"
                )

            if "chunk_id" not in candidate:
                raise ValueError(
                    "Candidate is missing chunk_id"
                )

            if "text" not in candidate:
                raise ValueError(
                    "Candidate is missing text"
                )

            if "score" not in candidate:
                raise ValueError(
                    "Candidate is missing retrieval score"
                )

            texts.append(
                candidate["text"]
            )

        scores = (
            self.reranker_provider.score(
                query,
                texts
            )
        )

        self.validate_scores(
            scores,
            expected_count=len(
                candidates
            )
        )

        reranked = []

        for candidate, rerank_score in zip(
            candidates,
            scores
        ):
            reranked.append(
                {
                    "chunk_id": (
                        candidate["chunk_id"]
                    ),
                    "text": candidate["text"],
                    "retrieval_score": (
                        candidate["score"]
                    ),
                    "rerank_score": (
                        float(rerank_score)
                    )
                }
            )

        reranked.sort(
            key=lambda item: (
                item["rerank_score"]
            ),
            reverse=True
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

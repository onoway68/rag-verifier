class FakeRerankerProvider:
    def __init__(
        self,
        scores=None,
        default_score=0.0
    ):
        self.scores = scores or {}
        self.default_score = float(
            default_score
        )

    def set_score(
        self,
        query,
        text,
        score
    ):
        self.scores[
            (query, text)
        ] = float(score)

    def score(
        self,
        query,
        texts
    ):
        if isinstance(texts, str):
            texts = [texts]

        return [
            self.scores.get(
                (query, text),
                self.default_score
            )
            for text in texts
        ]


class CrossEncoderRerankerProvider:
    def __init__(
        self,
        model_id,
        revision=None
    ):
        from sentence_transformers import (
            CrossEncoder
        )

        self.model = CrossEncoder(
            model_id,
            revision=revision
        )

    def score(
        self,
        query,
        texts
    ):
        if isinstance(texts, str):
            texts = [texts]

        pairs = [
            (query, text)
            for text in texts
        ]

        scores = self.model.predict(
            pairs
        )

        return [
            float(score)
            for score in scores
        ]

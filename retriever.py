import math


class Retriever:
    def __init__(
        self,
        chunks,
        embedding_provider
    ):
        self.chunks = chunks
        self.embedding_provider = embedding_provider

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):
        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have the same dimension"
            )

        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b
            )
        )

        norm_a = math.sqrt(
            sum(
                value * value
                for value in vector_a
            )
        )

        norm_b = math.sqrt(
            sum(
                value * value
                for value in vector_b
            )
        )

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (
            norm_a * norm_b
        )

    def retrieve(
        self,
        query,
        top_k=3,
        min_score=None
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        chunk_ids = list(
            self.chunks.keys()
        )

        chunk_texts = [
            self.chunks[chunk_id]
            for chunk_id in chunk_ids
        ]

        query_vector = (
            self.embedding_provider.encode(
                [query]
            )[0]
        )

        chunk_vectors = (
            self.embedding_provider.encode(
                chunk_texts
            )
        )

        results = []

        for (
            chunk_id,
            chunk_text,
            chunk_vector
        ) in zip(
            chunk_ids,
            chunk_texts,
            chunk_vectors
        ):
            score = self.cosine_similarity(
                query_vector,
                chunk_vector
            )

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "score": score
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
class FakeEmbeddingProvider:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}

    def set_embedding(self, text, vector):
        self.embeddings[text] = list(vector)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        return [
            list(self.embeddings[text])
            for text in texts
        ]


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_id=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        revision=None
    ):
        from sentence_transformers import (
            SentenceTransformer
        )

        self.model = SentenceTransformer(
            model_id,
            revision=revision
        )

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return [
            vector.tolist()
            for vector in embeddings
        ]
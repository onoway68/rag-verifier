from retriever import Retriever


class RoleAwareEmbeddingProvider:
    def __init__(self):
        self.query_calls = []
        self.document_calls = []

    def encode_queries(self, texts):
        self.query_calls.append(list(texts))
        return [[1.0, 0.0]]

    def encode_documents(self, texts):
        self.document_calls.append(list(texts))
        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]


def test_retriever_uses_query_embedding_role():
    provider = RoleAwareEmbeddingProvider()

    retriever = Retriever(
        chunks={
            "c1": "Relevant document.",
            "c2": "Other document.",
        },
        embedding_provider=provider,
    )

    retriever.retrieve(
        "medical query",
        top_k=2,
    )

    assert provider.query_calls == [
        ["medical query"]
    ]


def test_retriever_uses_document_embedding_role():
    provider = RoleAwareEmbeddingProvider()

    retriever = Retriever(
        chunks={
            "c1": "Relevant document.",
            "c2": "Other document.",
        },
        embedding_provider=provider,
    )

    retriever.retrieve(
        "medical query",
        top_k=2,
    )

    assert provider.document_calls == [
        [
            "Relevant document.",
            "Other document.",
        ]
    ]


def test_query_and_document_roles_are_not_interchanged():
    provider = RoleAwareEmbeddingProvider()

    retriever = Retriever(
        chunks={
            "c1": "Relevant document.",
            "c2": "Other document.",
        },
        embedding_provider=provider,
    )

    retriever.retrieve(
        "medical query",
        top_k=2,
    )

    assert provider.query_calls == [
        ["medical query"]
    ]

    assert provider.document_calls == [
        [
            "Relevant document.",
            "Other document.",
        ]
    ]


class RecordingSentenceTransformerModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append(
            ("encode", list(texts), normalize_embeddings)
        )
        return FakeArrayBatch(
            [[1.0, 0.0]]
        )

    def encode_query(
        self,
        texts,
        normalize_embeddings=True
    ):
        self.calls.append(
            (
                "encode_query",
                list(texts),
                normalize_embeddings,
            )
        )
        return FakeArrayBatch(
            [[1.0, 0.0]]
        )

    def encode_document(
        self,
        texts,
        normalize_embeddings=True
    ):
        self.calls.append(
            (
                "encode_document",
                list(texts),
                normalize_embeddings,
            )
        )
        return FakeArrayBatch(
            [[1.0, 0.0]]
        )


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeArrayBatch:
    def __init__(self, vectors):
        self.vectors = [
            FakeVector(vector)
            for vector in vectors
        ]

    def __iter__(self):
        return iter(self.vectors)


def make_sentence_transformer_provider(model):
    from embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    provider = (
        SentenceTransformerEmbeddingProvider.__new__(
            SentenceTransformerEmbeddingProvider
        )
    )
    provider.model = model
    return provider


def test_sentence_transformer_uses_role_aware_model_entry_points():
    model = RecordingSentenceTransformerModel()
    provider = make_sentence_transformer_provider(
        model
    )

    query_vectors = provider.encode_queries(
        ["query"]
    )
    document_vectors = provider.encode_documents(
        ["document"]
    )

    assert query_vectors == [[1.0, 0.0]]
    assert document_vectors == [[1.0, 0.0]]

    assert model.calls == [
        (
            "encode_query",
            ["query"],
            True,
        ),
        (
            "encode_document",
            ["document"],
            True,
        ),
    ]


def test_role_methods_preserve_legacy_encode_override():
    from embedding_provider import (
        SentenceTransformerEmbeddingProvider,
    )

    class LegacyCustomProvider(
        SentenceTransformerEmbeddingProvider
    ):
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(list(texts))
            return [[0.5, 0.5]]

    provider = LegacyCustomProvider()

    assert provider.encode_queries(
        ["query"]
    ) == [[0.5, 0.5]]

    assert provider.encode_documents(
        ["document"]
    ) == [[0.5, 0.5]]

    assert provider.calls == [
        ["query"],
        ["document"],
    ]

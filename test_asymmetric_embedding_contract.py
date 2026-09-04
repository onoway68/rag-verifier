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

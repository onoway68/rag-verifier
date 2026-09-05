from healthcare_retrieval_benchmark import HealthcareRetrievalBenchmark
from model_qualification import ModelQualificationRecordBuilder
from retriever import Retriever


class HealthcareEmbeddingQualificationRunner:
    """Run one governed healthcare retrieval qualification at a time."""

    def __init__(
        self,
        provider_factory,
        policy,
        benchmark_id,
        benchmark_version,
        top_k=5,
    ):
        if not callable(provider_factory):
            raise ValueError("provider_factory must be callable")

        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string")

        if (
            not isinstance(benchmark_version, str)
            or not benchmark_version.strip()
        ):
            raise ValueError("benchmark_version must be a non-empty string")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.provider_factory = provider_factory
        self.record_builder = ModelQualificationRecordBuilder(policy)
        self.benchmark_id = benchmark_id
        self.benchmark_version = benchmark_version
        self.top_k = top_k

    @staticmethod
    def _validate_model(model):
        if not isinstance(model, dict):
            raise ValueError("model must be a dict")

        required = {
            "model_id",
            "model_revision",
            "provider_type",
            "embedding_dimension",
        }

        if not required.issubset(model):
            raise ValueError("model metadata is incomplete")

        for field in ("model_id", "model_revision", "provider_type"):
            value = model[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")

        dimension = model["embedding_dimension"]
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer")

    @staticmethod
    def _validate_chunks(chunks):
        if not isinstance(chunks, dict) or not chunks:
            raise ValueError("chunks must be a non-empty dict")

        for chunk_id, text in chunks.items():
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("chunk ids must be non-empty strings")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("chunk text must be a non-empty string")

    @staticmethod
    def _validate_observed_embedding(provider, chunks, expected_dimension):
        encode_documents = getattr(provider, "encode_documents", None)
        if not callable(encode_documents):
            encode_documents = getattr(provider, "encode", None)

        if not callable(encode_documents):
            raise ValueError(
                "embedding provider must provide encode_documents() or encode()"
            )

        first_text = next(iter(chunks.values()))
        embeddings = encode_documents([first_text])

        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise ValueError("embedding provider returned an invalid probe batch")

        vector = embeddings[0]
        if not isinstance(vector, list) or not vector:
            raise ValueError("embedding provider returned an invalid probe vector")

        if len(vector) != expected_dimension:
            raise ValueError(
                "observed embedding dimension does not match model metadata"
            )

    def qualify(self, model, chunks, queries):
        self._validate_model(model)
        self._validate_chunks(chunks)

        provider = self.provider_factory(dict(model))
        self._validate_observed_embedding(
            provider,
            chunks,
            model["embedding_dimension"],
        )

        retriever = Retriever(chunks, provider)
        benchmark = HealthcareRetrievalBenchmark(
            retriever,
            top_k=self.top_k,
        )
        metrics = benchmark.evaluate(queries)

        benchmark_metadata = {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "top_k": self.top_k,
        }

        return self.record_builder.build(
            model=dict(model),
            benchmark=benchmark_metadata,
            metrics=metrics,
        )

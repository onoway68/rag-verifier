from retriever import Retriever
from retrieval_evaluator import RetrievalEvaluator


def build_chunk_corpus(
    documents,
    chunker
):
    chunks = {}

    for document_id, text in documents.items():
        document_chunks = chunker.chunk(
            text,
            document_id=document_id
        )

        for chunk in document_chunks:
            chunks[
                chunk["chunk_id"]
            ] = chunk["text"]

    return chunks


def chunk_id_to_document_id(
    chunk_id
):
    document_id, separator, index = (
        chunk_id.rpartition("-")
    )

    if (
        not separator
        or not document_id
        or not index.isdigit()
    ):
        raise ValueError(
            "Invalid chunk_id format"
        )

    return document_id


def evaluate_chunking_strategy(
    documents,
    queries,
    chunker,
    embedding_provider,
    top_k=3
):
    chunks = build_chunk_corpus(
        documents,
        chunker
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=(
            embedding_provider
        )
    )

    evaluator = RetrievalEvaluator()

    evaluated_queries = []
    dataset_cases = []

    for item in queries:
        results = retriever.retrieve(
            query=item["query"],
            top_k=top_k
        )

        retrieved_chunk_ids = [
            result["chunk_id"]
            for result in results
        ]

        retrieved_document_ids = [
            chunk_id_to_document_id(
                chunk_id
            )
            for chunk_id
            in retrieved_chunk_ids
        ]

        relevant_document_ids = list(
            item["relevant_document_ids"]
        )

        metrics = evaluator.evaluate_query(
            retrieved_ids=(
                retrieved_document_ids
            ),
            relevant_ids=(
                relevant_document_ids
            ),
            k=top_k
        )

        evaluated_queries.append(
            {
                "query": item["query"],
                "retrieved_chunk_ids": (
                    retrieved_chunk_ids
                ),
                "retrieved_document_ids": (
                    retrieved_document_ids
                ),
                "relevant_document_ids": (
                    relevant_document_ids
                ),
                "metrics": metrics
            }
        )

        dataset_cases.append(
            {
                "retrieved_ids": (
                    retrieved_document_ids
                ),
                "relevant_ids": (
                    relevant_document_ids
                )
            }
        )

    summary = evaluator.evaluate_dataset(
        cases=dataset_cases,
        k=top_k
    )

    return {
        "chunk_count": len(chunks),
        "summary": summary,
        "queries": evaluated_queries
    }

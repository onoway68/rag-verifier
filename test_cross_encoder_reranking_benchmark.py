import pytest

from chunker import WordChunker
from chunking_evaluator import (
    build_chunk_corpus
)
from embedding_provider import (
    SentenceTransformerEmbeddingProvider
)
from reranker import Reranker
from reranker_provider import (
    CrossEncoderRerankerProvider
)
from reranking_evaluator import (
    evaluate_reranking
)
from retriever import Retriever


EMBEDDING_MODEL_ID = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

EMBEDDING_MODEL_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f"
    "5529b4d41"
)

RERANKER_MODEL_ID = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)

RERANKER_MODEL_REVISION = (
    "233902d25c440f23af6f7d6e94d2946b"
    "ac0bee0a"
)


DOCUMENTS = {
    "hypertension": (
        "Hypertension is persistent elevation of arterial "
        "blood pressure and is an important modifiable "
        "cardiovascular risk factor. Many people have no "
        "obvious symptoms, so repeated accurate blood "
        "pressure measurement is important for detection. "
        "Long-term uncontrolled hypertension can damage "
        "blood vessels and contribute to coronary artery "
        "disease, heart failure, stroke, chronic kidney "
        "disease, and retinal injury. Management commonly "
        "includes lifestyle measures such as reducing "
        "excess dietary sodium, maintaining physical "
        "activity, controlling weight, and limiting harmful "
        "alcohol intake. Antihypertensive medications may "
        "also be required depending on blood pressure, "
        "comorbid conditions, and overall cardiovascular "
        "risk."
    ),
    "diabetes": (
        "Diabetes mellitus is characterized by chronic "
        "hyperglycemia caused by impaired insulin secretion, "
        "impaired insulin action, or both. Type 1 diabetes "
        "results from destruction of pancreatic beta cells "
        "and requires insulin replacement. Type 2 diabetes "
        "is strongly associated with insulin resistance and "
        "progressive beta-cell dysfunction. Persistent high "
        "glucose can injure both small and large blood "
        "vessels. Complications include retinopathy, kidney "
        "disease, peripheral neuropathy, cardiovascular "
        "disease, and impaired wound healing. Management "
        "includes glucose monitoring, nutrition, physical "
        "activity, appropriate medications, and systematic "
        "screening for complications and associated "
        "cardiovascular risk factors."
    )
}


QUERY = (
    "Which chronic metabolic disease can cause "
    "retinopathy, neuropathy, kidney disease, "
    "and impaired wound healing?"
)


@pytest.mark.integration
def test_cross_encoder_repairs_hard_minilm_ranking():
    chunker = WordChunker(
        chunk_size=40,
        overlap=0
    )

    chunks = build_chunk_corpus(
        documents=DOCUMENTS,
        chunker=chunker
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            model_id=EMBEDDING_MODEL_ID,
            revision=(
                EMBEDDING_MODEL_REVISION
            )
        )
    )

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=(
            embedding_provider
        )
    )

    reranker_provider = (
        CrossEncoderRerankerProvider(
            model_id=RERANKER_MODEL_ID,
            revision=(
                RERANKER_MODEL_REVISION
            )
        )
    )

    reranker = Reranker(
        reranker_provider
    )

    result = evaluate_reranking(
        retriever=retriever,
        reranker=reranker,
        cases=[
            {
                "query": QUERY,
                "relevant_ids": [
                    "diabetes-001"
                ]
            }
        ],
        candidate_k=5,
        final_k=1
    )

    case = result["cases"][0]

    print()
    print(
        "Candidates:",
        case["candidate_ids"]
    )
    print(
        "Baseline:",
        case["baseline_ids"]
    )
    print(
        "Reranked:",
        case["reranked_ids"]
    )
    print(
        "Baseline MRR:",
        result["baseline"]["mrr"]
    )
    print(
        "Reranked MRR:",
        result["reranked"]["mrr"]
    )

    assert (
        "diabetes-001"
        in case["candidate_ids"]
    )

    assert (
        case["baseline_ids"][0]
        == "hypertension-001"
    )

    assert (
        case["reranked_ids"][0]
        == "diabetes-001"
    )

    assert (
        result["baseline"]["mrr"]
        == 0.0
    )

    assert (
        result["reranked"]["mrr"]
        == 1.0
    )

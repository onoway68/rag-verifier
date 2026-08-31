import pytest

from chunker import WordChunker
from chunking_evaluator import (
    evaluate_chunking_strategy
)
from embedding_provider import (
    SentenceTransformerEmbeddingProvider
)


MODEL_ID = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

MODEL_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f"
    "5529b4d41"
)


DOCUMENTS = {
    "myocardial-infarction": (
        "Myocardial infarction occurs when blood flow "
        "through a coronary artery becomes severely "
        "reduced or completely blocked, causing ischemic "
        "injury to heart muscle. Atherosclerotic plaque "
        "rupture with formation of an acute thrombus is "
        "a common mechanism. Patients may develop chest "
        "pressure, shortness of breath, sweating, nausea, "
        "or pain radiating to the arm or jaw. Diagnosis "
        "uses the clinical presentation together with "
        "electrocardiographic findings and cardiac "
        "biomarkers such as troponin. Rapid restoration "
        "of coronary blood flow is important because "
        "prolonged ischemia increases myocardial damage "
        "and the risk of complications."
    ),

    "pneumonia": (
        "Pneumonia is an infection involving the lung "
        "parenchyma and alveolar spaces. It may be caused "
        "by bacteria, viruses, fungi, or other pathogens. "
        "Inflammation and accumulation of fluid or cellular "
        "material within alveoli can interfere with normal "
        "gas exchange. Common manifestations include cough, "
        "fever, shortness of breath, sputum production, "
        "fatigue, and pleuritic chest discomfort. Evaluation "
        "may include physical examination, pulse oximetry, "
        "chest imaging, and microbiologic testing when "
        "appropriate. Severity assessment helps determine "
        "whether a patient can be managed outside the "
        "hospital or requires more intensive monitoring "
        "and supportive treatment."
    ),

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
    ),

    "heart-failure": (
        "Heart failure is a clinical syndrome in which "
        "the heart cannot provide adequate circulation "
        "without abnormal elevation of filling pressures. "
        "It may occur with reduced or preserved left "
        "ventricular ejection fraction. Patients commonly "
        "experience exertional shortness of breath, fatigue, "
        "orthopnea, or peripheral edema. Neurohormonal "
        "activation and sodium and water retention can "
        "worsen congestion. Evaluation often incorporates "
        "history, physical examination, electrocardiography, "
        "laboratory testing, natriuretic peptides, and "
        "echocardiography. Treatment depends on the heart "
        "failure phenotype and underlying cause and may "
        "include diuretics for congestion together with "
        "therapies intended to improve symptoms, reduce "
        "hospitalization, or improve survival."
    ),

    "asthma": (
        "Asthma is a chronic inflammatory airway disorder "
        "associated with variable airflow obstruction and "
        "airway hyperresponsiveness. Symptoms can include "
        "episodic wheezing, cough, chest tightness, and "
        "shortness of breath. Triggers differ among patients "
        "and may include allergens, respiratory infections, "
        "exercise, smoke, occupational exposures, or cold "
        "air. Spirometry can demonstrate variable expiratory "
        "airflow limitation and may show improvement after "
        "bronchodilator administration. Long-term management "
        "aims to control symptoms, prevent exacerbations, "
        "and maintain normal activity. Inhaled anti-"
        "inflammatory therapy is central to management for "
        "many patients, while bronchodilators are used "
        "according to the treatment strategy and severity."
    )
}


QUERIES = [
    {
        "query": (
            "What causes myocardial injury when a "
            "coronary artery becomes blocked?"
        ),
        "relevant_document_ids": [
            "myocardial-infarction"
        ]
    },
    {
        "query": (
            "Which lung infection fills alveolar spaces "
            "and can interfere with gas exchange?"
        ),
        "relevant_document_ids": [
            "pneumonia"
        ]
    },
    {
        "query": (
            "Which condition can cause long-term kidney, "
            "retinal, stroke, and cardiovascular damage "
            "from elevated arterial pressure?"
        ),
        "relevant_document_ids": [
            "hypertension"
        ]
    },
    {
        "query": (
            "Which chronic metabolic disease can cause "
            "retinopathy, neuropathy, kidney disease, "
            "and impaired wound healing?"
        ),
        "relevant_document_ids": [
            "diabetes"
        ]
    },
    {
        "query": (
            "Which cardiac syndrome commonly causes "
            "orthopnea, edema, fatigue, and congestion?"
        ),
        "relevant_document_ids": [
            "heart-failure"
        ]
    },
    {
        "query": (
            "Which inflammatory airway disorder causes "
            "variable airflow obstruction, wheezing, "
            "and airway hyperresponsiveness?"
        ),
        "relevant_document_ids": [
            "asthma"
        ]
    }
]


CHUNKING_STRATEGIES = [
    (20, 0),
    (40, 0),
    (40, 10),
    (80, 20)
]


@pytest.mark.integration
def test_minilm_chunking_strategies():
    provider = (
        SentenceTransformerEmbeddingProvider(
            model_id=MODEL_ID,
            revision=MODEL_REVISION
        )
    )

    benchmark_results = []

    for chunk_size, overlap in CHUNKING_STRATEGIES:
        chunker = WordChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )

        result = evaluate_chunking_strategy(
            documents=DOCUMENTS,
            queries=QUERIES,
            chunker=chunker,
            embedding_provider=provider,
            top_k=3
        )

        summary = result["summary"]

        for query_result in result["queries"]:
            retrieved_documents = (
                query_result[
                    "retrieved_document_ids"
                ]
            )

            relevant_documents = set(
                query_result[
                    "relevant_document_ids"
                ]
            )

            first_document = (
                retrieved_documents[0]
                if retrieved_documents
                else None
            )

            if first_document not in relevant_documents:
                print(
                    "NON-RANK-1:",
                    f"chunk_size={chunk_size}",
                    f"overlap={overlap}",
                    query_result["query"],
                    "retrieved=",
                    retrieved_documents
                )

                print(
                    "retrieved_chunk_ids=",
                    query_result[
                        "retrieved_chunk_ids"
                    ]
                )

        benchmark_results.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunk_count": (
                    result["chunk_count"]
                ),
                "mean_precision_at_k": (
                    summary[
                        "mean_precision_at_k"
                    ]
                ),
                "mean_recall_at_k": (
                    summary[
                        "mean_recall_at_k"
                    ]
                ),
                "mrr": summary["mrr"]
            }
        )

        assert summary["query_count"] == 6

        assert (
            summary["mean_recall_at_k"]
            >= 0.75
        )

        assert summary["mrr"] >= 0.75

    print()
    print("MiniLM chunking benchmark")
    print("=" * 72)

    for item in benchmark_results:
        print(
            "chunk_size={size:<3} "
            "overlap={overlap:<3} "
            "chunks={count:<3} "
            "P@3={precision:.3f} "
            "R@3={recall:.3f} "
            "MRR={mrr:.3f}".format(
                size=item["chunk_size"],
                overlap=item["overlap"],
                count=item["chunk_count"],
                precision=(
                    item[
                        "mean_precision_at_k"
                    ]
                ),
                recall=(
                    item[
                        "mean_recall_at_k"
                    ]
                ),
                mrr=item["mrr"]
            )
        )
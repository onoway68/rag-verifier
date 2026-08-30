import json
from verifier import RAGVerifier


chunks = {

    "mi-001": (
        "Myocardial infarction occurs when blood flow to part "
        "of the heart muscle is blocked or severely reduced."
    ),

    "mi-002": (
        "Symptoms may include chest discomfort, shortness of "
        "breath, nausea, or sweating."
    )
}


retrieved_ids = {
    "mi-001",
    "mi-002"
}


answer = (
    "A heart attack occurs when blood flow to part of the "
    "heart muscle is blocked or severely reduced. [mi-001] "
    "Symptoms may include chest discomfort and sweating. [mi-002]"
)


print("Loading verifier...")

verifier = RAGVerifier(
    chunks=chunks,
    retrieved_ids=retrieved_ids
)


result = verifier.verify_answer(
    answer
)


print()
print("ANSWER:")
print(answer)

print()
print("STATUS:")
print(result["status"])

print()
print("REASON:")
print(result["reason"])

print()
print("TRUSTED RELEASE:")
print(result["trusted_release"])

print()
print("CITATION COVERAGE:")
print(
    result["citation_coverage"]["coverage_percent"],
    "%"
)

print()
print("JSON:")
print(
    json.dumps(
        result,
        indent=2
    )
)

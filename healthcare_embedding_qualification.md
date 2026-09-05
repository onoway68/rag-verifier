# Healthcare embedding qualification

This slice connects the governed healthcare retrieval benchmark to a concrete embedding-provider qualification run.

The runner:

- validates candidate model identity, pinned revision, provider type, and declared embedding dimension;
- probes the provider before benchmarking and rejects a dimension mismatch;
- executes the existing `HealthcareRetrievalBenchmark` through the production `Retriever` contract;
- evaluates the resulting metrics through the existing `ModelQualificationPolicy`; and
- emits the existing governed `ModelQualificationRecord`.

Operational model-loading or inference failures remain operational failures. They are not converted into a model `FAIL` qualification result, because `FAIL` is reserved for a completed benchmark whose metrics fail policy thresholds.

The deterministic tests use `FakeEmbeddingProvider`. Actual healthcare model candidates and externally sourced benchmark corpora remain a subsequent integration/evaluation step so that model downloads and network access do not enter the deterministic unit gate.

class RetrievalEvaluator:
    @staticmethod
    def validate_k(k):
        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

    @classmethod
    def precision_at_k(
        cls,
        retrieved_ids,
        relevant_ids,
        k
    ):
        cls.validate_k(k)

        retrieved_at_k = list(
            retrieved_ids[:k]
        )

        relevant_ids = set(
            relevant_ids
        )

        relevant_retrieved = len(
            set(retrieved_at_k)
            & relevant_ids
        )

        return (
            relevant_retrieved
            / k
        )

    @classmethod
    def recall_at_k(
        cls,
        retrieved_ids,
        relevant_ids,
        k
    ):
        cls.validate_k(k)

        relevant_ids = set(
            relevant_ids
        )

        if not relevant_ids:
            return 0.0

        retrieved_at_k = set(
            retrieved_ids[:k]
        )

        relevant_retrieved = len(
            retrieved_at_k
            & relevant_ids
        )

        return (
            relevant_retrieved
            / len(relevant_ids)
        )

    @staticmethod
    def reciprocal_rank(
        retrieved_ids,
        relevant_ids
    ):
        relevant_ids = set(
            relevant_ids
        )

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1
        ):
            if chunk_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def evaluate_query(
        self,
        retrieved_ids,
        relevant_ids,
        k
    ):
        self.validate_k(k)

        return {
            "precision_at_k": self.precision_at_k(
                retrieved_ids,
                relevant_ids,
                k
            ),
            "recall_at_k": self.recall_at_k(
                retrieved_ids,
                relevant_ids,
                k
            ),
            "reciprocal_rank": self.reciprocal_rank(
                retrieved_ids,
                relevant_ids
            )
        }

    def evaluate_dataset(
        self,
        cases,
        k
    ):
        self.validate_k(k)

        if not cases:
            return {
                "mean_precision_at_k": 0.0,
                "mean_recall_at_k": 0.0,
                "mrr": 0.0,
                "query_count": 0
            }

        results = [
            self.evaluate_query(
                retrieved_ids=case["retrieved_ids"],
                relevant_ids=case["relevant_ids"],
                k=k
            )
            for case in cases
        ]

        query_count = len(results)

        return {
            "mean_precision_at_k": sum(
                result["precision_at_k"]
                for result in results
            ) / query_count,
            "mean_recall_at_k": sum(
                result["recall_at_k"]
                for result in results
            ) / query_count,
            "mrr": sum(
                result["reciprocal_rank"]
                for result in results
            ) / query_count,
            "query_count": query_count
        }

    def evaluate_retriever(
        self,
        retriever,
        cases,
        k
    ):
        self.validate_k(k)

        evaluated_cases = []

        for case in cases:
            results = retriever.retrieve(
                query=case["query"],
                top_k=k
            )

            retrieved_ids = [
                result["chunk_id"]
                for result in results
            ]

            metrics = self.evaluate_query(
                retrieved_ids=retrieved_ids,
                relevant_ids=case["relevant_ids"],
                k=k
            )

            evaluated_cases.append(
                {
                    "query": case["query"],
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": case["relevant_ids"],
                    "metrics": metrics
                }
            )

        dataset_cases = [
            {
                "retrieved_ids": case[
                    "retrieved_ids"
                ],
                "relevant_ids": case[
                    "relevant_ids"
                ]
            }
            for case in evaluated_cases
        ]

        summary = self.evaluate_dataset(
            cases=dataset_cases,
            k=k
        )

        return {
            "queries": evaluated_cases,
            "summary": summary
        }

import re


class CitationAwareGenerator:
    def __init__(
        self,
        provider
    ):
        if provider is None:
            raise ValueError(
                "provider is required"
            )

        if not callable(
            getattr(
                provider,
                "generate",
                None
            )
        ):
            raise ValueError(
                "provider must define generate()"
            )

        self.provider = provider

    @staticmethod
    def extract_citations(text):
        return re.findall(
            r"\[([A-Za-z0-9_-]+)\]",
            text
        )

    @staticmethod
    def validate_question(question):
        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "question must be a non-empty string"
            )

    @staticmethod
    def validate_context(
        context,
        citation_map
    ):
        if not isinstance(context, str):
            raise ValueError(
                "context must be a string"
            )

        if not isinstance(
            citation_map,
            dict
        ):
            raise ValueError(
                "citation_map must be a dictionary"
            )

        if (
            not context.strip()
            or not citation_map
        ):
            raise ValueError(
                "grounding evidence is required"
            )

        for citation_id, text in citation_map.items():
            if (
                not isinstance(citation_id, str)
                or not citation_id.strip()
            ):
                raise ValueError(
                    "citation IDs must be non-empty strings"
                )

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(
                    "citation evidence must be a non-empty string"
                )

        context_citations = re.findall(
            r"\[([A-Za-z0-9_-]+)\]",
            context
        )

        if (
            set(context_citations)
            != set(citation_map.keys())
        ):
            raise ValueError(
                "context citation IDs must match citation_map"
            )

    @staticmethod
    def validate_answer(answer):
        if (
            not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(
                "provider must return a non-empty string"
            )

    def generate(
        self,
        question,
        context,
        citation_map
    ):
        self.validate_question(
            question
        )

        self.validate_context(
            context,
            citation_map
        )

        answer = self.provider.generate(
            question,
            context
        )

        self.validate_answer(
            answer
        )

        citation_ids = (
            self.extract_citations(
                answer
            )
        )

        unknown_citations = [
            citation_id
            for citation_id in citation_ids
            if citation_id not in citation_map
        ]

        if unknown_citations:
            raise ValueError(
                "Generated answer contains "
                "unknown citation IDs: "
                + ", ".join(
                    unknown_citations
                )
            )

        return {
            "answer": answer,
            "citation_ids": citation_ids
        }

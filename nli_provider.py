class HuggingFaceNLIProvider:
    """
    Real NLI inference provider backed by a Hugging Face
    sequence-classification model.

    PyTorch and Transformers are imported only when this
    provider is actually instantiated.
    """

    def __init__(
        self,
        model_id="cross-encoder/nli-deberta-v3-small"
    ):
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification
        )

        self.model_id = model_id
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id
        )

        self.model.eval()


    def predict(
        self,
        premise,
        hypothesis
    ):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True
        )

        with self._torch.no_grad():

            logits = self.model(
                **inputs
            ).logits

        probabilities = self._torch.softmax(
            logits,
            dim=-1
        )[0]

        scores = {}

        for index in range(
            len(probabilities)
        ):
            label = self.model.config.id2label[
                index
            ].lower()

            scores[label] = float(
                probabilities[index]
            )

        return scores


class FakeNLIProvider:
    """
    Deterministic NLI provider for unit testing.

    Does not import or execute PyTorch, Transformers,
    or any Hugging Face model.
    """

    def __init__(
        self,
        default_scores=None
    ):
        self.default_scores = (
            default_scores
            if default_scores is not None
            else {
                "contradiction": 0.01,
                "entailment": 0.98,
                "neutral": 0.01
            }
        )

        self.responses = {}


    def set_response(
        self,
        premise,
        hypothesis,
        scores
    ):
        self.responses[
            (premise, hypothesis)
        ] = dict(scores)


    def predict(
        self,
        premise,
        hypothesis
    ):
        return dict(
            self.responses.get(
                (premise, hypothesis),
                self.default_scores
            )
        )

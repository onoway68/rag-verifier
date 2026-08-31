class FakeGeneratorProvider:
    def __init__(self, response=""):
        self.response = response

    def generate(
        self,
        question,
        context
    ):
        return self.response


class HuggingFaceGeneratorProvider:
    def __init__(
        self,
        model_id="openai/gpt-oss-20b",
        token=None
    ):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(
            model=model_id,
            token=token
        )

    def generate(
        self,
        question,
        context
    ):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied "
                        "evidence. Cite supporting evidence "
                        "using citation markers such as "
                        "[C1] and [C2]."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Evidence:\n{context}"
                    )
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

import os

import pytest

from generator_provider import (
    HuggingFaceGeneratorProvider
)


@pytest.mark.integration
def test_huggingface_generator_returns_text():
    token = os.environ.get("HF_TOKEN")

    if not token:
        pytest.skip(
            "HF_TOKEN is not configured"
        )

    provider = HuggingFaceGeneratorProvider(
        model_id="openai/gpt-oss-20b",
        token=token
    )

    response = provider.generate(
        question=(
            "What complication of diabetes "
            "is described?"
        ),
        context=(
            "[C1] Diabetes can cause "
            "peripheral neuropathy."
        )
    )

    assert isinstance(
        response,
        str
    )

    assert response.strip()

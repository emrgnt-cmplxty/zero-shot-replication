"""A module for providing zero-shot completions from the Anthropic API."""
from zero_shot_replication.llm_providers.base import LargeLanguageModelProvider
from zero_shot_replication.model import AnthropicModel, ModelName, Quantization


class AnthropicZeroShotProvider(LargeLanguageModelProvider):
    """A concrete class to provide zero-shot completions from the Anthropic API."""

    MAX_TOKENS_TO_SAMPLE = (
        4_096  # This is a large value, we should check if it makes sense
    )

    def __init__(
        self,
        model_name: ModelName = ModelName.CLAUDE_2,
        quantization: Quantization = Quantization.proprietary,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> None:
        if quantization != Quantization.proprietary:
            raise ValueError(
                "Anthropic models only support proprietary quantization."
            )
        self._model = AnthropicModel(
            model_name,
            quantization,
            temperature,
            stream,
            AnthropicZeroShotProvider.MAX_TOKENS_TO_SAMPLE,
        )

    def get_completion(self, prompt: str) -> str:
        """Get a completion from the Anthropic provider based on the provided prompt."""
        return self.model.get_completion(prompt)

    @property
    def model(self) -> AnthropicModel:
        return self._model

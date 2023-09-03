"""A module for providing zero-shot completions from the OpenAI API."""
import logging

from zero_shot_replication.llm_providers.base import LargeLanguageModelProvider
from zero_shot_replication.model import ModelName, OpenAIModel, Quantization

logger = logging.getLogger(__name__)


class OpenAIZeroShotProvider(LargeLanguageModelProvider):
    """A class to provide zero-shot completions from the OpenAI API."""

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization = Quantization.proprietary,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> None:
        if quantization != Quantization.proprietary:
            raise ValueError(
                "Anthropic models only support proprietary quantization."
            )

        self._model = OpenAIModel(
            model_name,
            quantization,
            temperature,
            stream,
        )

    def get_completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API based on the provided prompt."""
        logger.info(
            f"Getting completion from OpenAI API for model={self.model.model_name}"
        )
        return self.model.get_completion(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    @property
    def model(self) -> OpenAIModel:
        return self._model

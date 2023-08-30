"""A module for providing zero-shot completions from the OpenAI API."""
import logging

from zero_shot_replication.llm_providers.base import LargeLanguageModelProvider
from zero_shot_replication.model import (
    HuggingFaceModel,
    ModelName,
    Quantization,
)

logger = logging.getLogger(__name__)


class HuggingFaceZeroShotProvider(LargeLanguageModelProvider):
    """A class to provide zero-shot completions from the OpenAI API."""

    def __init__(
        self,
        model: ModelName,
        quantization: Quantization = Quantization.float16,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> None:
        if quantization == Quantization.proprietary:
            raise ValueError(
                "HuggingFace models do not support proprietary quantization."
            )

        self._model = HuggingFaceModel(
            model, quantization, temperature, stream
        )

    def get_completion(self, prompt: str) -> str:
        """Get a completion from the Local HuggingFace API based on the provided prompt."""
        logger.info(
            f"Getting completion from Local HuggingFace API for model={self.model.model_name}"
        )
        return self.model.get_completion(prompt)

    @property
    def model(self) -> HuggingFaceModel:
        return self._model

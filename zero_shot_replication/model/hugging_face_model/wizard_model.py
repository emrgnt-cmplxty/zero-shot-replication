import logging

import torch

from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
    Quantization,
)

logger = logging.getLogger(__name__)


class HuggingFaceWizardModel(LargeLanguageModel):
    """A class to provide zero-shot completions from a local Llama model."""

    # TODO - Make these upstream configurations?
    MAX_NEW_TOKENS = 1_024
    TOP_K = 40
    TOP_P = 0.9
    NUM_BEAMS = 1
    VERSION = "0.1.0"

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
        max_new_tokens=None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ValueError(
                "Project must be installed with optional package vllm to run WizardCoder."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Selecting device = {self.device}")
        super().__init__(
            model_name,
            quantization,
            temperature,
            stream,
            prompt_mode=PromptMode.HUMAN_FEEDBACK,
        )

        # TODO - Introduce multi-gpu support
        self.model = LLM(model=model_name.value, tensor_parallel_size=1)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1,
            max_tokens=HuggingFaceWizardModel.MAX_NEW_TOKENS,
        )
        self.temperature = temperature

    def get_completion(self, prompt: str) -> str:
        """Generate the completion from the Wizard model."""
        with torch.no_grad():
            completions = self.model.generate([prompt], self.sampling_params)
        gen_seq = completions[0].outputs[0].text
        return gen_seq.split(prompt)[-1]

import logging

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, __version__

from zero_shot_replication.core.utils import quantization_to_kwargs
from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
    Quantization,
)

logger = logging.getLogger(__name__)


class HuggingFaceCodeLlamaModel(LargeLanguageModel):
    """A class to provide zero-shot completions from a local Llama model."""

    # TODO - Make these upstream configurations?
    MAX_TOTAL_TOKENS = 4_096
    MAX_NEW_TOKENS = 1_024
    TOP_K = 40
    TOP_P = 0.75
    DO_SAMPLE = True
    VERSION = "0.1.0"

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Selecting device = {self.device}")

        # TODO - It is possible that the main branch might not be in dev mode
        # this check can be made more robust.
        if "dev" not in __version__:
            raise ValueError(
                f"Installed from PyPI with version: {__version__}"
            )

        super().__init__(
            model_name,
            quantization,
            temperature,
            stream,
            prompt_mode=PromptMode.HUMAN_FEEDBACK,
        )
        # TODO - Add support for 4-bit
        self.model = LlamaForCausalLM.from_pretrained(
            model_name.value,
            device_map="auto",
            **quantization_to_kwargs(quantization),
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name.value,
            device_map="auto",
            **quantization_to_kwargs(quantization),
        )

    def get_completion(self, prompt: str) -> str:
        """Generate the completion from the Phind model."""

        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=HuggingFaceCodeLlamaModel.MAX_TOTAL_TOKENS,
        ).to(self.device)

        # Generate
        generate_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=HuggingFaceCodeLlamaModel.MAX_NEW_TOKENS,
            do_sample=HuggingFaceCodeLlamaModel.DO_SAMPLE,
            top_p=HuggingFaceCodeLlamaModel.TOP_P,
            top_k=HuggingFaceCodeLlamaModel.TOP_K,
            temperature=self.temperature,
        )
        completion = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        completion = completion.replace(prompt, "")
        return completion

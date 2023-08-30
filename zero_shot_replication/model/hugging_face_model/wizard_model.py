import logging

import torch
from transformers import GenerationConfig, __version__

from zero_shot_replication.core.utils import quantization_to_kwargs
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
    TRANSFORMERS_VERSION = "4.32.0"
    VERSION = "0.1.0"

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
        max_new_tokens=None,
    ) -> None:
        if HuggingFaceWizardModel.TRANSFORMERS_VERSION != __version__:
            raise ValueError(
                f"Transformers version is not correct, {HuggingFaceWizardModel.TRANSFORMERS_VERSION} was expected, but {__version__} was found."
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

        try:
            from vllm import LLM, SamplingParams
        except:
            raise ValueError("vllm must be installed to run Wizard models.")

        # TODO - Introduce multi-gpu support
        self.model = LLM(base_model=model_name.value, tensor_parallel_size=1)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1,
            max_tokens=HuggingFaceWizardModel.MAX_NEW_TOKENS,
        )
        self.temperature = temperature

    def get_completion(self, prompt: str) -> str:
        """Generate the completion from the Wizard model."""
        with torch.no_grad():
            completions = self.llm.generate([prompt], self.sampling_params)
        gen_seqs = completions[0].outputs[0].text
        print("gen_seqs = ", gen_seqs)
        return gen_seqs[0].split("### Response:")[-1]

        # inputs = self.tokenizer(
        #     prompt, return_tensors="pt", truncation=True, padding=True
        # ).to(self.device)
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # generation_config = GenerationConfig(
        #     temperature=self.temperature,
        #     top_p=HuggingFaceWizardModel.TOP_P,
        #     top_k=HuggingFaceWizardModel.TOP_K,
        #     num_beams=HuggingFaceWizardModel.NUM_BEAMS,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     do_sample=True,
        # )

        # generate_ids = self.model.generate(
        #     inputs["input_ids"],
        #     generation_config=generation_config,
        #     max_new_tokens=self.max_new_tokens,
        # )
        # completion = self.tokenizer.batch_decode(
        #     generate_ids, skip_special_tokens=True
        # )[0]

        # if prompt in completion:
        #     completion = completion.split(prompt)[1]
        # return completion

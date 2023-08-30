import logging

from zero_shot_replication.core.utils import quantization_to_kwargs
from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
    Quantization,
)

logger = logging.getLogger(__name__)


class HuggingFacePhindModel(LargeLanguageModel):
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
        try:
            import torch
            from transformers_git import AutoTokenizer, LlamaForCausalLM
        except ImportError:
            raise ValueError(
                "Project must be installed with optional package transformers_git to run Phind."
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

        self.model = LlamaForCausalLM.from_pretrained(
            model_name.value,
            device_map="auto",
            **quantization_to_kwargs(quantization),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
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
            max_length=HuggingFacePhindModel.MAX_TOTAL_TOKENS,
        ).to(self.device)

        # Generate
        generate_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=HuggingFacePhindModel.MAX_NEW_TOKENS,
            do_sample=HuggingFacePhindModel.DO_SAMPLE,
            top_p=HuggingFacePhindModel.TOP_P,
            top_k=HuggingFacePhindModel.TOP_K,
            temperature=self.temperature,
        )
        completion = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        completion = completion.replace(prompt, "").split("\n\n\n")[0]
        return completion

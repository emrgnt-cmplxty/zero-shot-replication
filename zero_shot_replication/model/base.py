from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from zero_shot_replication.core import BaseDataset, PromptMode


class Quantization(Enum):
    proprietary = "proprietary"
    float16 = "float16"
    bfloat16 = "bfloat16"
    four_bit = "four-bit"
    eight_bit = "eight-bit"


class ModelName(Enum):
    """An enum to hold the names of the models."""

    # OpenAI Models

    ## GPT-3.5
    GPT_3p5_TURBO_0301 = "gpt-3.5-turbo-0301"
    GPT_3p5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3p5_TURBO = "gpt-3.5-turbo"

    ## GPT-4
    GPT_4_0314 = "gpt-4-0314"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4 = "gpt-4"

    # Anthropic Models
    CLAUDE_INSTANT_1 = "claude-instant-1"
    CLAUDE_2 = "claude-2"

    # Meta Open Source Models
    LLAMA_2_7B_HF = "meta-llama/Llama-2-7b-hf"
    LLAMA_2_13B_HF = "meta-llama/Llama-2-13b-hf"
    LLAMA_2_70B_HF = "meta-llama/Llama-2-70b-hf"
    CODE_LLAMA_7B_HF = "codellama/CodeLlama-7b-hf"
    CODE_LLAMA_13B_HF = "codellama/CodeLlama-13b-hf"
    CODE_LLAMA_34B_HF = "codellama/CodeLlama-34b-hf"
    CODE_LLAMA_7B_PYTHON_HF = "codellama/CodeLlama-7b-Python-hf"
    CODE_LLAMA_13B_PYTHON_HF = "codellama/CodeLlama-13b-Python-hf"
    CODE_LLAMA_34B_PYTHON_HF = "codellama/CodeLlama-34b-Python-hf"

    # Meta Open Source Models local (weights need to be downloaded)
    CODE_LLAMA_7B_PYTHON = "CodeLlama-7b-Python"
    CODE_LLAMA_13B_PYTHON = "CodeLlama-13b-Python"
    CODE_LLAMA_34B_PYTHON = "CodeLlama-34b-Python"

    # Other HF Open Source Models
    WIZARD_LM_PYTHON_34B = "WizardLM/WizardCoder-Python-34B-V1.0"
    PHIND_LM_PYTHON_34B = "Phind/Phind-CodeLlama-34B-v1"
    PHIND_LM_PYTHON_34B_V2 = "Phind/Phind-CodeLlama-34B-v2"


class LargeLanguageModel(ABC):
    """An abstract class to provide a common interface for LLMs."""

    VERSION = "0.1.0"

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
        prompt_mode: PromptMode,
    ) -> None:
        self.model_name = model_name
        self.quantization = quantization
        self.temperature = temperature
        self.stream = stream
        self.prompt_mode = prompt_mode

    @abstractmethod
    def get_completion(self, input: Any) -> str:
        """Abstract method to get a completion from the provider."""
        pass

    def get_formatted_prompt(self, problem: dict, dataset: BaseDataset) -> str:
        """Default concrete method to get a formatted prompt for the provider."""
        return dataset.get_formatted_prompt(problem, self.prompt_mode)

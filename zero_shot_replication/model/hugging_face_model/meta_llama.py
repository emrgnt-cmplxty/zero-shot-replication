import logging
import os

import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    StoppingCriteria,
)

from zero_shot_replication.core.utils import quantization_to_kwargs
from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
    Quantization,
)

logger = logging.getLogger(__name__)


class HuggingFaceLlamaModel(LargeLanguageModel):
    MAX_TOTAL_TOKENS = 4_096
    MAX_NEW_TOKENS = 1_024
    TOP_K = 40
    TOP_P = 0.75
    DO_SAMPLE = True
    # VERSION = "0.1.0" version shouldn't be needed in here

    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Selecting device = {self.device}")

        super().__init__(
            model_name,
            quantization,
            temperature,
            stream,
            prompt_mode=PromptMode.COMPLETION,
        )
        assert quantization == Quantization.bfloat16, "Please use bfloat16"

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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=HuggingFaceLlamaModel.MAX_TOTAL_TOKENS,
        ).to(self.device)

        # Generate
        generate_ids = self.model.generate(
            inputs,
            max_new_tokens=HuggingFaceLlamaModel.MAX_NEW_TOKENS,
            do_sample=HuggingFaceLlamaModel.DO_SAMPLE,
            top_p=HuggingFaceLlamaModel.TOP_P,
            top_k=HuggingFaceLlamaModel.TOP_K,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        completion = self.tokenizer.batch_decode(
            generate_ids[:, len(inputs[0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        completion = prompt + completion.split("\n\n\n")[0]
        return completion


HUMANEVAL_EOS = [
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
    "\n\n\n\n\n",
]
NON_CODE_EOS = [
    "<|endoftext|>",
    "\n```",
    "\n</s>",
    "<|endofmask|>",
    "</s>",
    "<EOT>",
]
EOS = HUMANEVAL_EOS + NON_CODE_EOS


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


CODE_LLAMA_ROOT = os.environ.get("CODE_LLAMA_ROOT", "/JawTitan/codellama/")


# S1: Install package from https://github.com/facebookresearch/codellama S2: Install model to ${CODE_LLAMA_ROOT} (
# This can be any actual path) S3: CODE_LLAMA_ROOT=?? torchrun --nproc_per_node 1 codegen/generate.py --model
# code-llama-7b --bs 1 --temperature 0 --n_samples 1 --resume --greedy
class LocalLlamaModel(LargeLanguageModel):
    MAX_TOTAL_TOKENS = 4_096
    MAX_NEW_TOKENS = 1_024
    TOP_K = 40
    TOP_P = 0.75
    DO_SAMPLE = True

    # VERSION = "0.1.0" version shouldn't be needed in here
    def __init__(
        self,
        model_name: ModelName,
        quantization: Quantization,
        temperature: float,
        stream: bool,
    ) -> None:
        assert CODE_LLAMA_ROOT is not None
        from llama import (  # See https://github.com/facebookresearch/codellama
            Llama,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Selecting device = {self.device}")

        super().__init__(
            model_name,
            quantization,
            temperature,
            stream,
            prompt_mode=PromptMode.COMPLETION,
        )

        self.generator = Llama.build(
            ckpt_dir=os.path.join(CODE_LLAMA_ROOT, model_name.value),
            tokenizer_path=os.path.join(
                CODE_LLAMA_ROOT, model_name.value, "tokenizer.model"
            ),
            max_seq_len=LocalLlamaModel.MAX_TOTAL_TOKENS,
            max_batch_size=1,
        )
        self.temperature = temperature

    @staticmethod
    def sanitize(gen_str: str):
        tmp = ""
        for line in str.splitlines(gen_str):
            lspace = len(line) - len(line.lstrip())
            if lspace == 3:
                tmp += " "
            tmp += line + "\n"
        new_code = tmp
        return new_code

    def get_completion(self, prompt: str) -> str:
        gen_strs = self.generator.text_completion(
            [prompt],
            max_gen_len=LocalLlamaModel.MAX_NEW_TOKENS,
            temperature=self.temperature,
            top_p=LocalLlamaModel.TOP_P,
            # top_k=LocalLlamaModel.TOP_K, # local model actually doesn't support top_k
        )
        gen_str = [gen_str["generation"] for gen_str in gen_strs][0]

        min_index = 10000
        for eos in EOS:
            if eos in gen_str:
                # could be multiple eos in outputs, better pick minimum one
                min_index = min(min_index, gen_str.index(eos))

        return self.sanitize(prompt + gen_str[:min_index])

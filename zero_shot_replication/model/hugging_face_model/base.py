from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
    Quantization,
)
from zero_shot_replication.model.hugging_face_model.hf_code_llama import (
    HuggingFaceCodeLlamaModel,
)
from zero_shot_replication.model.hugging_face_model.meta_llama import (
    LocalLlamaModel,
)
from zero_shot_replication.model.hugging_face_model.phind_model import (
    HuggingFacePhindModel,
)
from zero_shot_replication.model.hugging_face_model.wizard_model import (
    HuggingFaceWizardModel,
)


class HuggingFaceModel(LargeLanguageModel):
    """A concrete class for creating OpenAI models."""

    META_MODELS = [
        ModelName.LLAMA_2_7B_HF,
        ModelName.LLAMA_2_13B_HF,
        ModelName.LLAMA_2_70B_HF,
        ModelName.CODE_LLAMA_7B_HF,
        ModelName.CODE_LLAMA_13B_HF,
        ModelName.CODE_LLAMA_34B_HF,
        ModelName.CODE_LLAMA_7B_PYTHON_HF,
        ModelName.CODE_LLAMA_13B_PYTHON_HF,
        ModelName.CODE_LLAMA_34B_PYTHON_HF,
    ]

    LOCAL_MODELS = [
        ModelName.CODE_LLAMA_7B_PYTHON,
        ModelName.CODE_LLAMA_13B_PYTHON,
        ModelName.CODE_LLAMA_34B_PYTHON,
    ]

    def __init__(
        self,
        model_name: ModelName = ModelName.GPT_4,
        quantization: Quantization = Quantization.float16,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> None:
        if stream:
            raise ValueError(
                "Stream is not supported for HuggingFace in this framework."
            )
        super().__init__(
            model_name,
            quantization,
            temperature,
            stream,
            prompt_mode=PromptMode.HUMAN_FEEDBACK,
        )

        if model_name in HuggingFaceModel.META_MODELS:
            self.model: LargeLanguageModel = HuggingFaceCodeLlamaModel(
                model_name,
                quantization,
                temperature,
                stream,
            )
        elif model_name in HuggingFaceModel.LOCAL_MODELS:
            self.model = LocalLlamaModel(
                model_name,
                quantization,
                temperature,
                stream,
            )
        elif model_name == ModelName.WIZARD_LM_PYTHON_34B:
            self.model = HuggingFaceWizardModel(
                model_name,
                quantization,
                temperature,
                stream,
            )
        elif model_name in [
            ModelName.PHIND_LM_PYTHON_34B,
            ModelName.PHIND_LM_PYTHON_34B_V2,
        ]:
            self.model = HuggingFacePhindModel(
                model_name,
                quantization,
                temperature,
                stream,
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

        # need to reset the prompt mode to the actual model used
        # there is two layer of models (one is this current wrapper file, and the other actual model)
        self.prompt_mode = self.model.prompt_mode

    def get_completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API based on the provided messages."""
        return self.model.get_completion(prompt)

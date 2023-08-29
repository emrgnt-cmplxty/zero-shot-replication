from zero_shot_replication.model.base import (
    LargeLanguageModel,
    ModelName,
    PromptMode,
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
        ModelName.CODE_LLAMA_7B,
        ModelName.CODE_LLAMA_13B,
        ModelName.CODE_LLAMA_34B,
    ]

    def __init__(
        self,
        model_name: ModelName = ModelName.GPT_4,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> None:
        if stream:
            raise ValueError(
                "Stream is not supported for HuggingFace in this framework."
            )
        super().__init__(
            model_name,
            temperature,
            stream,
            prompt_mode=PromptMode.HUMAN_FEEDBACK,
        )

        if model_name in HuggingFaceModel.META_MODELS:
            raise NotImplementedError("Meta models are not supported yet.")
        elif model_name == ModelName.WIZARD_LM_PYTHON_34B:
            self.model: LargeLanguageModel = HuggingFaceWizardModel(
                model_name,
                temperature,
                stream,
            )
        elif model_name == ModelName.PHIND_LM_PYTHON_34B:
            self.model = HuggingFacePhindModel(
                model_name,
                temperature,
                stream,
            )

    def get_completion(self, prompt: str) -> str:
        """Get a completion from the OpenAI API based on the provided messages."""
        return self.model.get_completion(prompt)

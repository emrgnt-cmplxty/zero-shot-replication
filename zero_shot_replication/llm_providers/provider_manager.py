from zero_shot_replication.llm_providers.anthropic_provider import (
    AnthropicZeroShotProvider,
)
from zero_shot_replication.llm_providers.automata_provider import (
    AutomataZeroShotProvider,
)
from zero_shot_replication.llm_providers.base import (
    MODEL_SETS,
    LargeLanguageModelProvider,
    ProviderConfig,
    ProviderName,
)
from zero_shot_replication.llm_providers.hugging_face_provider import (
    HuggingFaceZeroShotProvider,
)
from zero_shot_replication.llm_providers.openai_provider import (
    OpenAIZeroShotProvider,
)
from zero_shot_replication.model import ModelName, Quantization


class ProviderManager:
    AUTOMATA_MODELS = [
        model
        for model in MODEL_SETS[ProviderName.OPENAI]
        if model
        not in [ModelName.GPT_3p5_TURBO_0301, ModelName.GPT_3p5_TURBO_0613]
    ]

    PROVIDERS = [
        ProviderConfig(
            ProviderName.OPENAI,
            MODEL_SETS[ProviderName.OPENAI],
            OpenAIZeroShotProvider,
        ),
        ProviderConfig(
            ProviderName.ANTHROPIC,
            MODEL_SETS[ProviderName.ANTHROPIC],
            AnthropicZeroShotProvider,
        ),
        ProviderConfig(
            ProviderName.HUGGING_FACE,
            MODEL_SETS[ProviderName.HUGGING_FACE],
            HuggingFaceZeroShotProvider,
        ),
        ProviderConfig(
            ProviderName.AUTOMATA,
            AUTOMATA_MODELS,
            AutomataZeroShotProvider,
        ),
    ]

    @staticmethod
    def get_provider(
        provider_name: ProviderName,
        model_name: ModelName,
        quantization: Quantization,
        *args,
        **kwargs,
    ) -> LargeLanguageModelProvider:
        for provider in ProviderManager.PROVIDERS:
            if (
                provider.name == provider_name
                and model_name not in provider.models
            ):
                raise ValueError(
                    f"Model '{model_name}' not supported by provider '{provider_name}'."
                )
            elif provider.name == provider_name:
                return provider.llm_class(
                    model_name, quantization, *args, **kwargs
                )

        raise ValueError(f"Provider '{provider_name}' not supported.")

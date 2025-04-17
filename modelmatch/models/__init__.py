import logging 
from .base import LLM
from .providers.openai import OpenAIModel
from .providers.google import GoogleModel
from .providers.open_router import OpenRouterModel
# Import other providers here as they are created
# from .providers.gemini import GeminiModel
# from .providers.mistral import MistralModel

# Get logger
logger = logging.getLogger(__name__)
# Add more models here as needed
_SUPPORTED_MODELS = {
    #"gpt-4o": OpenAIModel,
    #"gpt-3.5-turbo": OpenAIModel,
    "meta-llama/llama-3.3-70b-instruct:free":OpenRouterModel,
    "mistralai/mistral-small-24b-instruct-2501:free":OpenRouterModel,
    "gemini-2.0-flash-thinking-exp":GoogleModel
}

def get_model(model_name: str) -> LLM:
    """Factory function to get an instance of a supported LLM."""
    logger.debug(f"Attempting to get model instance for: {model_name}")
    model_name_lower = model_name.lower()
    model_class = _SUPPORTED_MODELS.get(model_name_lower)

    if model_class:
        try:
            model_instance = model_class(model_id=model_name_lower)
            logger.info(f"Successfully initialized model: {model_name}")
            return model_instance
        except ValueError as e: # Catch config errors from provider __init__
             logger.error(f"Configuration error for model '{model_name}': {e}")
             raise ValueError(f"Configuration error for model '{model_name}': {e}")
        except Exception as e:
            logger.error(f"Failed to initialize model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model '{model_name}': {e}")
    else:
        logger.error(f"Unsupported model requested: '{model_name}'. Supported: {list(_SUPPORTED_MODELS.keys())}")
        raise ValueError(f"Unsupported model: '{model_name}'. Supported models are: {list(_SUPPORTED_MODELS.keys())}")

def list_available_models() -> list[str]:
    """Returns a list of available model names."""
    return list(_SUPPORTED_MODELS.keys())
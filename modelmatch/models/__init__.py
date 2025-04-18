import logging
import yaml
import os
from typing import Dict, Type, List, Tuple, Optional

from .base import LLM
# Import all necessary provider classes
from .providers.openai import OpenAIModel
from .providers.open_router import OpenRouterModel
from .providers.google import GoogleModel

logger = logging.getLogger(__name__)

# Map provider string names (from YAML) to actual Class objects
PROVIDER_MAP: Dict[str, Type[LLM]] = {
    "OpenAIModel": OpenAIModel,
    "OpenRouterModel": OpenRouterModel,
    "GoogleModel": GoogleModel,
}

# Load models configuration from YAML
_MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_config.yaml')
# Stores: model_id -> (display_name, ProviderClass)
_SUPPORTED_MODELS_INFO: Dict[str, Tuple[str, Type[LLM]]] = {}
# Stores: display_name -> model_id (for reverse lookup from CLI)
_DISPLAY_NAME_TO_MODEL_ID: Dict[str, str] = {}

try:
    logger.info(f"Loading model configuration from: {_MODEL_CONFIG_PATH}")
    if not os.path.exists(_MODEL_CONFIG_PATH):
        logger.error(f"Model configuration file not found: {_MODEL_CONFIG_PATH}")
        raise FileNotFoundError(f"Model configuration file not found: {_MODEL_CONFIG_PATH}")

    with open(_MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    if not config_data or 'models' not in config_data or not isinstance(config_data['models'], list):
        logger.error(f"Invalid format in {_MODEL_CONFIG_PATH}. Expected a list under the 'models' key.")
        raise ValueError(f"Invalid format in {_MODEL_CONFIG_PATH}")

    loaded_count = 0
    for model_entry in config_data['models']:
        model_id = model_entry.get('model_id')
        display_name = model_entry.get('display_name', model_id)
        provider_name = model_entry.get('provider')

        if not model_id or not provider_name:
            logger.warning(f"Skipping invalid model entry in YAML (missing id or provider): {model_entry}")
            continue

        provider_class = PROVIDER_MAP.get(provider_name)
        if not provider_class:
            logger.warning(f"Skipping model '{model_id}': Provider class '{provider_name}' not found or mapped in PROVIDER_MAP.")
            continue

        # Store main info
        if model_id in _SUPPORTED_MODELS_INFO:
            logger.warning(f"Duplicate model_id '{model_id}' found in YAML. Overwriting.")
        _SUPPORTED_MODELS_INFO[model_id] = (display_name, provider_class)

        # --- Store reverse mapping ---
        if display_name in _DISPLAY_NAME_TO_MODEL_ID:
            logger.warning(f"Duplicate display_name '{display_name}' found in YAML. CLI mapping will only point to the last model_id ('{model_id}'). Ensure display names are unique if needed.")
        _DISPLAY_NAME_TO_MODEL_ID[display_name] = model_id
        # --- End reverse mapping ---

        loaded_count += 1

    logger.info(f"Successfully loaded {loaded_count} model configurations from YAML.")

except FileNotFoundError:
    # Error already logged, _SUPPORTED_MODELS_INFO will be empty
    pass
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML file {_MODEL_CONFIG_PATH}: {e}", exc_info=True)
    # _SUPPORTED_MODELS_INFO will be empty or partially filled
    pass
except Exception as e:
     logger.error(f"Unexpected error loading model config: {e}", exc_info=True)


def get_model(model_id: str) -> LLM:
    """
    Factory function to get an instance of a supported LLM based on loaded config.
    Expects the actual model_id, not the display name.
    """
    logger.debug(f"Attempting to get model instance for actual ID: {model_id}")

    if model_id not in _SUPPORTED_MODELS_INFO:
        # This should ideally be caught in main.py after name translation
        logger.error(f"Model ID '{model_id}' not found in supported models info.")
        raise ValueError(f"Model ID '{model_id}' not found. Available IDs: {list_available_models()}")

    display_name, model_class = _SUPPORTED_MODELS_INFO[model_id]
    logger.debug(f"Found config for '{model_id}': Display Name='{display_name}', Provider={model_class.__name__}")

    try:
        model_instance = model_class(model_id=model_id)
        logger.info(f"Successfully initialized model: {model_id} ({display_name})")
        return model_instance
    # ... (keep existing exception handling for initialization) ...
    except ValueError as e: # Catch config errors from provider __init__ (e.g., missing API key)
         logger.error(f"Configuration error initializing model '{model_id}': {e}")
         # Re-raise with more context if helpful
         raise ValueError(f"Configuration error for model '{model_id} ({display_name})': {e}") from e
    except Exception as e:
        logger.error(f"Failed to initialize model instance '{model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize model instance '{model_id} ({display_name})': {e}")


def list_available_models() -> List[str]:
    """Returns a list of available model IDs loaded from config."""
    return list(_SUPPORTED_MODELS_INFO.keys())

def list_available_models_display() -> List[Tuple[str, str]]:
    """Returns a list of (display_name, model_id) tuples."""
    # Sort alphabetically by display name for cleaner help text
    return sorted(
        [(info[0], model_id) for model_id, info in _SUPPORTED_MODELS_INFO.items()],
        key=lambda item: item[0] # Sort by display name
    )

def get_model_id_from_display_name(name: str) -> Optional[str]:
    """
    Looks up a model ID by its display name. Case-sensitive.

    Args:
        name: The display name to look up.

    Returns:
        The corresponding model ID string, or None if not found.
    """
    return _DISPLAY_NAME_TO_MODEL_ID.get(name)

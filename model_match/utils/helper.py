import json
import logging # Import logging
from typing import Any, Dict, List

# Get a logger for this module
logger = logging.getLogger(__name__)

def load_json_data(filepath: str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    logger.debug(f"Attempting to load JSON data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully decoded JSON from {filepath}")

        # Validation
        if "prompt_template" not in data or "data" not in data:
             msg = "Input JSON must contain 'prompt_template' and 'data' keys."
             logger.error(f"{msg} File: {filepath}")
             raise ValueError(msg)
        if not isinstance(data["data"], list):
             msg = "'data' key must contain a list of data points."
             logger.error(f"{msg} File: {filepath}")
             raise ValueError(msg)

        logger.info(f"Loaded JSON data successfully from {filepath}.")
        return data
    except FileNotFoundError:
        logger.error(f"Input file not found at {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from {filepath}: {e}", exc_info=True)
        raise
    except ValueError as e:
        # Error already logged during validation check
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}", exc_info=True)
        raise


def format_prompt(template: str, data_point: Any) -> str:
    """Formats the prompt template with the given data point."""
    logger.debug(f"Formatting prompt template with data: {str(data_point)[:100]}...")
    try:
        if isinstance(data_point, dict):
             formatted = template.format(**data_point)
        else:
             try:
                 formatted = template.format(data=data_point)
             except KeyError:
                 logger.warning(f"Could not find '{{data}}' placeholder in template when formatting with non-dict data. Using template directly. Data: {data_point}")
                 formatted = template # Return template as is, maybe log warning

        logger.debug(f"Prompt formatted successfully (length: {len(formatted)}).")
        return formatted
    except KeyError as e:
        logger.error(f"Missing key '{e}' in data point {data_point} needed for prompt template: '{template}'", exc_info=True)
        raise ValueError(f"Missing key '{e}' in data point needed for prompt template") from e
    except Exception as e:
        logger.error(f"Error formatting prompt template '{template}' with data '{data_point}': {e}", exc_info=True)
        raise ValueError(f"Failed to format prompt template") from e
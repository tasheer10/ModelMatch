import logging
import openai
from modelmatch.models.base import LLM
from modelmatch.config import settings

# Get a logger for this module
logger = logging.getLogger(__name__)

class OpenRouterModel(LLM):
    """Concrete implementation for OpenRouter models."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        logger.debug(f"Initializing OpenAIModel for model_id: {model_id}")
        if not settings.OPEN_ROUTER_API_KEY:
            logger.error("OPEN_ROUTER_API_KEY not found in environment settings.")
            raise ValueError("OPEN_ROUTER_API_KEY not found in environment settings.")

        # Recommended: Initialize the client instance once
        try:
            if(settings.OPEN_ROUTER_API_KEY == None):
                raise Exception("Open Router API Key Not Set")
            self.client = openai.OpenAI(base_url = "https://openrouter.ai/api/v1",
                                        api_key=settings.OPENAI_API_KEY)
            logger.debug("Open Router client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Open Router client: {e}", exc_info=True)
            raise

    def generate(self, prompt: str) -> str:
        """Generates text using the specified OpenAI model."""
        logger.debug(f"Generating text with Open Router model: {self.model_id}")
        try:
            # Using ChatCompletion API 
            chat_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = chat_response.choices[0].message.content.strip()
            logger.debug(f"Open Router response received (length: {len(response_text)} chars).")
            # Log token usage if needed (and available in response)
            if usage := getattr(chat_response, 'usage', None):
                 logger.debug(f"Open Router API usage for {self.model_id}: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")
            return response_text

        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            logger.error(f"Open Router API Error for model {self.model_id}: {e}", exc_info=True)
            raise # Re-raise APIError or a custom exception
        except openai.RateLimitError as e:
             logger.error(f"Open Router Rate Limit Exceeded for model {self.model_id}: {e}", exc_info=True)
             raise
        except openai.AuthenticationError as e:
             logger.error(f"Open Router Authentication Error: {e}. Check your API key.", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"An unexpected error occurred with Open Router model {self.model_id}: {e}", exc_info=True)
            raise
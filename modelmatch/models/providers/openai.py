import logging # Import logging
import openai
from modelmatch.models.base import LLM
from modelmatch.config import settings

# Get a logger for this module
logger = logging.getLogger(__name__)

class OpenAIModel(LLM):
    """Concrete implementation for OpenAI models."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        logger.debug(f"Initializing OpenAIModel for model_id: {model_id}")
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment settings.")
            raise ValueError("OPENAI_API_KEY not found in environment settings.")

        # Recommended: Initialize the client instance once
        try:
            if(settings.OPENAI_API_KEY == None):
                raise Exception("Open AI API Key Not Set")
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.debug("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    def generate(self, prompt: str) -> str:
        """Generates text using the specified OpenAI model."""
        logger.debug(f"Generating text with OpenAI model: {self.model_id}")
        try:
            # Using ChatCompletion API 
            chat_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = chat_response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response received (length: {len(response_text)} chars).")
            # Log token usage if needed (and available in response)
            if usage := getattr(chat_response, 'usage', None):
                 logger.debug(f"OpenAI API usage for {self.model_id}: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")
            return response_text

        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            logger.error(f"OpenAI API Error for model {self.model_id}: {e}", exc_info=True)
            raise # Re-raise APIError or a custom exception
        except openai.RateLimitError as e:
             logger.error(f"OpenAI Rate Limit Exceeded for model {self.model_id}: {e}", exc_info=True)
             raise
        except openai.AuthenticationError as e:
             logger.error(f"OpenAI Authentication Error: {e}. Check your API key.", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"An unexpected error occurred with OpenAI model {self.model_id}: {e}", exc_info=True)
            raise
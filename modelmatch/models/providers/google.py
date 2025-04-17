import logging
from google import genai
from modelmatch.models.base import LLM
from modelmatch.config import settings

# Get a logger for this module
logger = logging.getLogger(__name__)

class GoogleModel(LLM):
    """Concrete implementation for Google models."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        logger.debug(f"Initializing Google API for model_id: {model_id}")
        if not settings.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not found in environment settings.")
            raise ValueError("GOOGLE_API_KEY not found in environment settings.")

        try:
            if(settings.GOOGLE_API_KEY == None):
                raise Exception("Google API Key Not Set")
            self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            logger.debug("Google Client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}", exc_info=True)
            raise

    def generate(self, prompt: str) -> str:
        """Generates text using the specified OpenAI model."""
        logger.debug(f"Generating text with Open Router model: {self.model_id}")
        try:
            # Using ChatCompletion API 
            chat_response = self.client.models.generate_content(
                                                                    model=self.model_id,
                                                                    contents=prompt,
                                                                )
            response_text = chat_response.text
            logger.debug(f"Google API Response response received (length: {len(response_text)} chars).")
            return response_text
        except Exception as e:
            logger.error(f"An unexpected error occurred with Open Router model {self.model_id}: {e}", exc_info=True)
            raise
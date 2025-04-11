from abc import ABC, abstractmethod

class LLM(ABC):
    """Abstract Base Class for Large Language Models."""

    def __init__(self, model_id: str):
        """
        Initializes the LLM provider.

        Args:
            model_id: A unique identifier for the specific model instance
                      (e.g., "gpt-4o", "gemini-1.5-pro").
        """
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated text response from the model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Can raise specific exceptions related to API calls,
                       authentication, rate limits, etc.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"LLM Provider ({self.__class__.__name__}, Model: {self.model_id})"
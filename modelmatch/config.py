import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings:
    """Loads settings from environment variables."""
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    OPEN_ROUTER_API_KEY: str | None = os.getenv("OPEN_ROUTER_API_KEY")

    # You could add other configurations here, like default models, timeouts etc.

# Create a single instance for easy access
settings = Settings()

# Basic validation (optional but recommended)
def check_config():
    if not settings.OPENAI_API_KEY: # Example check
        print("Warning: OPENAI_API_KEY not found in environment variables or .env file.")
    # Add checks for other required keys as needed

# Run checks when the module is imported (optional)
# check_config()
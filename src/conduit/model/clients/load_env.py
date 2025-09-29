from dotenv import load_dotenv
import os


def load_env(api_key: str) -> str:
    """
    Load any .env files that exist, then load api_key from environment.
    If user has no api_key, return an error.
    """
    # Load .env file it exists
    load_dotenv()
    # Get API key with env var taking precedence
    found_api_key = os.getenv(api_key)
    if not found_api_key:
        raise ValueError(f"{api_key} not found in environment or .env file.")
    else:
        return found_api_key

"""
Chalna Settings - Configuration via environment variables or .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vibevoice_api_url: str = "http://localhost:8001"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

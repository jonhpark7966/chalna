"""
Chalna Settings - Configuration via environment variables or .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vibevoice_model_path: str = "microsoft/VibeVoice-ASR"
    sentry_dsn: str = ""
    sentry_environment: str = "local"
    sentry_release: str = ""
    sentry_traces_sample_rate: float = 0.05

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

"""Settings from the environment or .env. Swapping the LLM means changing LLM_* values only."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Any OpenAI-compatible chat completions endpoint. The default is the local dev model.
    llm_base_url: str = "http://localhost:3636/v1"
    llm_model: str = "google/gemma-4-e2b"
    llm_api_key: SecretStr = SecretStr("not-needed")  # local servers ignore it
    llm_temperature: float = 0.2
    llm_timeout: float = 120.0  # seconds; the local model takes about 25s per call

    agent_max_tool_rounds: int = 6

    mock_data_dir: Path = REPO_ROOT / "data" / "mock"

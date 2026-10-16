from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM selection
    llm_provider: str = Field(default="gemini", description="gemini | openai | groq")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    # None uses each provider's default; "json_schema" is strictest where the model supports it.
    llm_structured_output_method: Literal["function_calling", "json_schema", "json_mode"] | None = None

    @field_validator("llm_structured_output_method", mode="before")
    @classmethod
    def _blank_means_default(cls, value: object) -> object:
        return value or None

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Tavily
    tavily_api_key: str = ""

    # Email
    sendgrid_api_key: str = ""
    sender_email: str = ""

    # Self-evaluation: documents scoring below the threshold get revised, up to max_revisions times
    review_quality_threshold: int = Field(default=80, ge=0, le=100)
    max_revisions: int = Field(default=1, ge=0, le=3)

    # Storage
    faiss_index_path: str = "data/faiss_index"
    max_resume_size_mb: int = Field(default=5, gt=0)

    # Logging
    log_level: str = "INFO"


settings = Settings()

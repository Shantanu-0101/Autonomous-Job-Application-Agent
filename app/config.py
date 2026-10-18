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
    gemini_model: str = "gemini-3.6-flash"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Tavily
    tavily_api_key: str = ""

    # Email (SMTP): works with Gmail/Outlook app passwords or SendGrid's relay (username "apikey")
    smtp_host: str = ""
    smtp_port: int = 587  # 587 = STARTTLS, 465 = implicit TLS
    smtp_username: str = ""
    smtp_password: str = ""
    sender_email: str = ""
    sender_name: str = ""

    # Exported documents
    pdf_page_size: Literal["a4", "letter"] = "a4"

    # Self-evaluation: documents scoring below the threshold get revised, up to max_revisions times
    review_quality_threshold: int = Field(default=80, ge=0, le=100)
    max_revisions: int = Field(default=1, ge=0, le=3)

    # Storage
    tracker_db_path: str = "data/tracker.sqlite"
    checkpoint_db_path: str = "data/checkpoints.sqlite"
    faiss_index_path: str = "data/faiss_index"
    max_resume_size_mb: int = Field(default=5, gt=0)

    # Memory
    research_cache_days: int = Field(default=14, ge=0)  # 0 disables the company research cache
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local FastEmbed model, ~70 MB, no API key
    similar_job_threshold: float = Field(default=0.75, gt=0, le=1)
    duplicate_job_threshold: float = Field(default=0.93, gt=0, le=1)

    # Logging
    log_level: str = "INFO"


settings = Settings()

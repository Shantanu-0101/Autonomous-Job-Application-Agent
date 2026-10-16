import pytest

from app.config import Settings


@pytest.fixture
def mock_settings():
    return Settings(
        llm_provider="gemini",
        google_api_key="test-key-not-real",
        tavily_api_key="test-tavily-key",
        log_level="DEBUG",
    )

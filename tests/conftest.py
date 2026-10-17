import pytest

from app.config import Settings, settings


@pytest.fixture(autouse=True)
def default_settings(monkeypatch):
    """Run every test against default settings with fake keys, ignoring the developer's .env."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    defaults = Settings(
        google_api_key="test-google-key",
        openai_api_key="test-openai-key",
        groq_api_key="test-groq-key",
        tavily_api_key="test-tavily-key",
    )
    for name in Settings.model_fields:
        monkeypatch.setattr(settings, name, getattr(defaults, name))

import pytest

from app.config import Settings


def test_default_provider():
    s = Settings(google_api_key="x")
    assert s.llm_provider == "gemini"


def test_custom_provider():
    s = Settings(llm_provider="openai", openai_api_key="x")
    assert s.llm_provider == "openai"


def test_default_temperature():
    s = Settings(google_api_key="x")
    assert s.llm_temperature == 0.1


def test_temperature_bounds():
    s = Settings(google_api_key="x", llm_temperature=0.5)
    assert 0.0 <= s.llm_temperature <= 2.0


def test_temperature_too_high():
    with pytest.raises(Exception):
        Settings(google_api_key="x", llm_temperature=3.0)


def test_default_model_names(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    s = Settings(google_api_key="x", _env_file=None)
    assert s.gemini_model == "gemini-2.5-flash"
    assert s.openai_model == "gpt-4o"
    assert s.groq_model == "llama-3.3-70b-versatile"


def test_faiss_default_path():
    s = Settings(google_api_key="x")
    assert s.faiss_index_path == "data/faiss_index"

from unittest.mock import patch, MagicMock

import pytest

from app.config import Settings
from app.llm.provider import get_llm


def test_unknown_provider_raises():
    cfg = Settings(llm_provider="unknown", google_api_key="x")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm(config=cfg)


def test_missing_gemini_key_raises():
    cfg = Settings(llm_provider="gemini", google_api_key="")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
        get_llm(config=cfg)


def test_missing_openai_key_raises():
    cfg = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        get_llm(config=cfg)


def test_missing_groq_key_raises():
    cfg = Settings(llm_provider="groq", groq_api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
        get_llm(config=cfg)


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_gemini_provider_constructed(mock_cls):
    cfg = Settings(llm_provider="gemini", google_api_key="fake-key")
    get_llm(config=cfg)
    mock_cls.assert_called_once()


@patch("langchain_openai.ChatOpenAI")
def test_openai_provider_constructed(mock_cls):
    cfg = Settings(llm_provider="openai", openai_api_key="fake-key")
    get_llm(config=cfg)
    mock_cls.assert_called_once()


@patch("langchain_groq.ChatGroq")
def test_groq_provider_constructed(mock_cls):
    cfg = Settings(llm_provider="groq", groq_api_key="fake-key")
    get_llm(config=cfg)
    mock_cls.assert_called_once()


@patch("langchain_openai.ChatOpenAI")
def test_provider_override_argument(mock_cls):
    cfg = Settings(
        llm_provider="gemini",
        google_api_key="x",
        openai_api_key="y",
    )
    get_llm(provider="openai", config=cfg)
    mock_cls.assert_called_once()


@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_temperature_forwarded(mock_cls):
    cfg = Settings(llm_provider="gemini", google_api_key="fake-key")
    get_llm(config=cfg)
    call_kwargs = mock_cls.call_args
    assert call_kwargs.kwargs["temperature"] == 0.1


def _fake_llm(output):
    from unittest.mock import MagicMock

    chain = MagicMock()
    chain.with_retry.return_value = chain
    chain.invoke.return_value = output
    llm = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm, chain


@patch("app.llm.provider.get_llm")
def test_invoke_structured_returns_model_with_retry(mock_get_llm):
    from app.llm.models import LLMHealthCheck
    from app.llm.provider import invoke_structured

    expected = LLMHealthCheck(status="ok", provider="groq", message="hi")
    mock_get_llm.return_value, chain = _fake_llm(expected)

    assert invoke_structured(LLMHealthCheck, "hello") is expected
    chain.with_retry.assert_called_once_with(stop_after_attempt=3)


@pytest.mark.parametrize("bad_output", [None, {"status": "ok"}])
@patch("app.llm.provider.get_llm")
def test_invoke_structured_rejects_invalid_output(mock_get_llm, bad_output):
    from app.llm.models import LLMHealthCheck
    from app.llm.provider import StructuredOutputError, invoke_structured

    mock_get_llm.return_value, _ = _fake_llm(bad_output)

    with pytest.raises(StructuredOutputError, match="LLMHealthCheck"):
        invoke_structured(LLMHealthCheck, "hello")

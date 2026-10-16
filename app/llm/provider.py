import logging
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.config import Settings, settings

logger = logging.getLogger("job_agent.llm")

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The LLM did not return data matching the requested schema."""


def invoke_structured(schema: type[T], prompt: str, *, retries: int = 4) -> T:
    method = settings.llm_structured_output_method
    base = get_llm()
    llm = base.with_structured_output(schema, method=method) if method else base.with_structured_output(schema)
    # Backoff (~4s, 8s, 16s) outlasts free-tier per-minute rate limits.
    chain = llm.with_retry(
        stop_after_attempt=retries,
        exponential_jitter_params={"initial": 4, "max": 30},
    )
    result = chain.invoke(prompt)
    if not isinstance(result, schema):
        raise StructuredOutputError(f"LLM returned no valid {schema.__name__}")
    return result


def get_llm(
    provider: str | None = None,
    config: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    cfg = config or settings
    provider = (provider or cfg.llm_provider).lower().strip()

    common = {"temperature": kwargs.pop("temperature", cfg.llm_temperature)}
    common.update(kwargs)

    if provider == "gemini":
        if not cfg.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini provider")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.gemini_model,
            google_api_key=cfg.google_api_key,
            **common,
        )

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.openai_model,
            api_key=cfg.openai_api_key,
            **common,
        )

    if provider == "groq":
        if not cfg.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq provider")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=cfg.groq_model,
            api_key=cfg.groq_api_key,
            **common,
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. Supported: gemini, openai, groq"
    )

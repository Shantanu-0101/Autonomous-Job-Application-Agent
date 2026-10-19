import re
import zlib
from collections.abc import Sequence

import numpy as np
import pytest

import app.memory.job_memory as job_memory
from app.config import Settings, settings
from app.memory.job_memory import JobMemory


def fake_embed(texts: Sequence[str]) -> np.ndarray:
    """Deterministic bag-of-words vectors: shared words mean similar texts. No model download."""
    vectors = np.full((len(texts), 256), 1e-6, dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vectors[row, zlib.crc32(word.encode()) % 256] += 1
    return vectors


@pytest.fixture(autouse=True)
def default_settings(monkeypatch, tmp_path):
    """Run every test against default settings with fake keys, ignoring the developer's .env,
    and with storage in a fresh temporary directory."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    defaults = Settings(
        google_api_key="test-google-key",
        openai_api_key="test-openai-key",
        groq_api_key="test-groq-key",
        tavily_api_key="test-tavily-key",
        tracker_db_path=str(tmp_path / "tracker.sqlite"),
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        faiss_index_path=str(tmp_path / "faiss_index"),
    )
    for name in Settings.model_fields:
        monkeypatch.setattr(settings, name, getattr(defaults, name))
    monkeypatch.setattr(
        job_memory, "_memory_at", lambda index_dir, *_model: JobMemory(index_dir, fake_embed)
    )

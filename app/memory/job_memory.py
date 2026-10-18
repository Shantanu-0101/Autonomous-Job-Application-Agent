import logging
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.llm.models import JobPosting

logger = logging.getLogger("job_agent.memory")

Embedder = Callable[[Sequence[str]], np.ndarray]


def job_text(job: JobPosting) -> str:
    """What makes two postings "the same job": who, what, and what they ask for."""
    parts = [f"{job.title} at {job.company}", job.description, *job.requirements, *job.responsibilities]
    return "\n".join(p for p in parts if p)[:4000]


def fastembed_embedder(model_name: str) -> Embedder:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name)  # downloads ~70 MB on first use, then runs locally on CPU

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.array(list(model.embed(list(texts))), dtype="float32")

    return embed


class JobMemory:
    """Semantic index of past job postings, keyed by application id (FAISS, cosine similarity)."""

    def __init__(self, index_dir: str | Path, embed: Embedder):
        self.index_dir = Path(index_dir)
        self.embed = embed
        self._lock = threading.Lock()
        self._index: faiss.Index | None = None
        index_file = self.index_dir / "jobs.faiss"
        if index_file.exists():
            self._index = faiss.read_index(str(index_file))

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.ascontiguousarray(self.embed(texts), dtype="float32")
        faiss.normalize_L2(vectors)  # inner product of unit vectors = cosine similarity
        return vectors

    def add(self, application_id: int, text: str) -> None:
        vector = self._vectors([text])
        ids = np.array([application_id], dtype="int64")
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector.shape[1]))
            # Re-recording a job replaces its old vector. FAISS's Python wrapper accepts an id array
            # here; its stubs only declare the C++ IDSelector form.
            self._index.remove_ids(ids)  # type: ignore[arg-type]
            self._index.add_with_ids(vector, ids)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_dir / "jobs.faiss"))

    def search(self, text: str, k: int = 5, min_score: float = 0.0) -> list[tuple[int, float]]:
        """(application_id, similarity) pairs, most similar first."""
        if not len(self):
            return []
        assert self._index is not None
        with self._lock:
            scores, ids = self._index.search(self._vectors([text]), min(k, len(self)))
        return [
            (int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1 and s >= min_score
        ]


@lru_cache(maxsize=2)
def _memory_at(index_dir: str, model_name: str) -> JobMemory:
    return JobMemory(index_dir, fastembed_embedder(model_name))


def get_job_memory() -> JobMemory | None:
    """The shared memory, or None if the embedding model can't be loaded (e.g. offline)."""
    try:
        return _memory_at(settings.faiss_index_path, settings.embedding_model)
    except Exception as e:
        logger.warning("Semantic job memory unavailable: %s", e)
        return None

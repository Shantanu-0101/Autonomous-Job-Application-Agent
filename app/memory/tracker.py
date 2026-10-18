import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel

from app.config import settings
from app.llm.models import CompanyResearch, JobPosting

ApplicationStatus = Literal[
    "ready",  # approved; the user applies on the job's page
    "sent",  # emailed by the agent
    "skipped",  # the user decided not to apply
    "applied",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
]
STATUSES: tuple[ApplicationStatus, ...] = (
    "ready", "sent", "skipped", "applied", "interviewing", "offer", "rejected", "withdrawn"
)
# Re-applying after these would mean sending the same company a second application.
SUBMITTED_STATUSES = {"sent", "applied", "interviewing", "offer", "rejected"}

_TRACKING_PARAMS = {"gh_src", "ref", "refid", "source", "src", "trk", "lever-source", "lever-origin"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_url TEXT NOT NULL DEFAULT '',
    normalized_url TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    company_key TEXT NOT NULL,
    title_key TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    fit_score INTEGER,
    submitted_to TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_applications_url ON applications (normalized_url);
CREATE INDEX IF NOT EXISTS idx_applications_role ON applications (company_key, title_key);
CREATE TABLE IF NOT EXISTS company_research (
    company_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


class ApplicationRecord(BaseModel):
    id: int
    job_url: str
    company: str
    title: str
    location: str
    status: ApplicationStatus
    fit_score: int | None
    submitted_to: str
    notes: str
    created_at: str
    updated_at: str


def normalize_job_url(url: str) -> str:
    """Same posting, same key: drop tracking params, case, scheme, and trailing slashes."""
    parsed = urlparse(url.strip())
    if not parsed.netloc:
        return ""
    query = [
        (k, v) for k, v in parse_qsl(parsed.query)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)
    ]
    host = parsed.netloc.lower().removeprefix("www.")
    return urlunparse(("https", host, parsed.path.rstrip("/"), "", urlencode(sorted(query)), ""))


def text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Tracker:
    """Application history and cached company research in a local SQLite database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:  # commits on success, rolls back on error
                yield conn

    @staticmethod
    def _record(row: sqlite3.Row) -> ApplicationRecord:
        return ApplicationRecord.model_validate({k: row[k] for k in row.keys()})

    # --- Applications ---

    def find(self, job_url: str = "", company: str = "", title: str = "") -> ApplicationRecord | None:
        """The most recent application for the same posting URL or the same company and title."""
        url_key = normalize_job_url(job_url) if job_url else ""
        company_key, title_key = text_key(company), text_key(title)
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM applications
                WHERE (? != '' AND normalized_url = ?)
                   OR (? != '' AND ? != '' AND company_key = ? AND title_key = ?)
                ORDER BY updated_at DESC, id DESC LIMIT 1
                """,
                (url_key, url_key, company_key, title_key, company_key, title_key),
            ).fetchone()
        return self._record(row) if row else None

    def get(self, application_id: int) -> ApplicationRecord | None:
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM applications WHERE id = ?", (application_id,)).fetchone()
        return self._record(row) if row else None

    def record(
        self,
        job: JobPosting,
        status: ApplicationStatus,
        fit_score: int | None = None,
        submitted_to: str = "",
    ) -> ApplicationRecord:
        """Insert an application, or update the existing record for the same job."""
        existing = self.find(job.source_url, job.company, job.title)
        now = _now()
        with self._transaction() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE applications SET status = ?, fit_score = COALESCE(?, fit_score),
                        submitted_to = CASE WHEN ? != '' THEN ? ELSE submitted_to END,
                        location = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, fit_score, submitted_to, submitted_to, job.location, now, existing.id),
                )
                application_id = existing.id
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO applications (job_url, normalized_url, company, title, company_key,
                        title_key, location, status, fit_score, submitted_to, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.source_url, normalize_job_url(job.source_url), job.company, job.title,
                        text_key(job.company), text_key(job.title), job.location, status, fit_score,
                        submitted_to, now, now,
                    ),
                )
                application_id = int(cursor.lastrowid or 0)
        record = self.get(application_id)
        assert record is not None
        return record

    def update_status(
        self, application_id: int, status: ApplicationStatus, notes: str | None = None
    ) -> ApplicationRecord:
        if status not in STATUSES:
            raise ValueError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}")
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE applications SET status = ?, notes = COALESCE(?, notes), updated_at = ? WHERE id = ?",
                (status, notes, _now(), application_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"No application with id {application_id}")
        record = self.get(application_id)
        assert record is not None
        return record

    def list(self, status: ApplicationStatus | None = None, limit: int = 500) -> list[ApplicationRecord]:
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM applications WHERE ? IS NULL OR status = ? "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (status, status, limit),
            ).fetchall()
        return [self._record(r) for r in rows]

    # --- Company research cache ---

    def cached_research(self, company: str, max_age_days: int) -> CompanyResearch | None:
        if max_age_days <= 0:
            return None
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT data, fetched_at FROM company_research WHERE company_key = ?",
                (text_key(company),),
            ).fetchone()
        if not row:
            return None
        if datetime.fromisoformat(row["fetched_at"]) < datetime.now(UTC) - timedelta(days=max_age_days):
            return None
        return CompanyResearch.model_validate(json.loads(row["data"]))

    def save_research(self, company: str, research: CompanyResearch) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO company_research (company_key, data, fetched_at) VALUES (?, ?, ?) "
                "ON CONFLICT (company_key) DO UPDATE SET data = excluded.data, fetched_at = excluded.fetched_at",
                (text_key(company), research.model_dump_json(), _now()),
            )


@lru_cache(maxsize=4)
def _tracker_at(path: str) -> Tracker:
    return Tracker(path)


def get_tracker() -> Tracker:
    return _tracker_at(settings.tracker_db_path)

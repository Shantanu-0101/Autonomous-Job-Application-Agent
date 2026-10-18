import logging

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.models import JobPosting
from app.memory.job_memory import JobMemory, job_text
from app.memory.tracker import (
    SUBMITTED_STATUSES,
    ApplicationRecord,
    ApplicationStatus,
    Tracker,
    text_key,
)

logger = logging.getLogger("job_agent.history")


class SimilarApplication(BaseModel):
    application: ApplicationRecord
    similarity: float


class HistoryCheck(BaseModel):
    duplicate: ApplicationRecord | None = None  # same posting URL, or same company and title
    possible_duplicate: SimilarApplication | None = None  # same company, near-identical posting
    similar: list[SimilarApplication] = Field(default_factory=list)
    blocked: bool = False  # already submitted — applying again needs explicit permission
    message: str = ""


def _when(record: ApplicationRecord) -> str:
    return record.updated_at[:10]


def check_history(job: JobPosting, tracker: Tracker, memory: JobMemory | None) -> HistoryCheck:
    duplicate = tracker.find(job.source_url, job.company, job.title)
    possible: SimilarApplication | None = None
    similar: list[SimilarApplication] = []

    if memory is not None:
        try:
            matches = memory.search(job_text(job), k=6, min_score=settings.similar_job_threshold)
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            matches = []
        for application_id, score in matches:
            record = tracker.get(application_id)
            if record is None or (duplicate and record.id == duplicate.id):
                continue
            match = SimilarApplication(application=record, similarity=round(score, 3))
            same_company = text_key(record.company) == text_key(job.company)
            if possible is None and same_company and score >= settings.duplicate_job_threshold:
                possible = match
            else:
                similar.append(match)

    check = HistoryCheck(duplicate=duplicate, possible_duplicate=possible, similar=similar)
    if duplicate:
        check.blocked = duplicate.status in SUBMITTED_STATUSES
        verb = "already applied to" if check.blocked else "looked at"
        check.message = (
            f"You {verb} {duplicate.title} at {duplicate.company} "
            f"(status: {duplicate.status}, {_when(duplicate)})."
        )
    elif possible:
        found = possible.application
        check.message = (
            f'This looks like "{found.title}" at {found.company}, which you marked {found.status} '
            f"on {_when(found)} ({possible.similarity:.0%} similar)."
        )
    return check


def record_application(
    job: JobPosting,
    status: ApplicationStatus,
    tracker: Tracker,
    memory: JobMemory | None,
    fit_score: int | None = None,
    submitted_to: str = "",
) -> ApplicationRecord:
    record = tracker.record(job, status, fit_score=fit_score, submitted_to=submitted_to)
    if memory is not None:
        try:
            memory.add(record.id, job_text(job))
        except Exception as e:  # the tracker row is the source of truth; the index is best-effort
            logger.warning("Could not add application %d to semantic memory: %s", record.id, e)
    return record

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Inputs — job_description (pasted text) takes priority over job_url
    resume_path: str
    job_url: str
    job_description: str

    # Module 2: Scraped job data
    job_details: dict[str, Any] | None

    # Module 3: Company research
    company_research: dict[str, Any] | None

    # Module 4: Parsed resume (pre-fill parsed_resume to skip parsing)
    resume_text: str | None
    parsed_resume: dict[str, Any] | None
    resume_warnings: list[str]

    # Module 5: Tailored resume (a TailoredResume dump)
    tailored_resume: dict[str, Any] | None
    tailoring_feedback: list[str]

    # Module 6: Cover letter (a CoverLetter dump)
    cover_letter_tone: str
    cover_letter: dict[str, Any] | None

    # Module 7: Evaluation (FitAssessment and ApplicationReview dumps)
    fit_assessment: dict[str, Any] | None
    application_review: dict[str, Any] | None
    # Revised documents awaiting review; accepted only if they don't score lower
    candidate_tailored_resume: dict[str, Any] | None
    candidate_cover_letter: dict[str, Any] | None
    revision_count: int
    max_revisions: int  # overrides settings.max_revisions when set

    # Module 8: Human approval
    approval_status: str  # "approved", "changes_requested", or "rejected"
    human_feedback: dict[str, list[str]] | None  # {"resume": [...], "cover_letter": [...]}
    user_requested_revision: bool  # pending candidates come from the user's feedback

    # Module 9: Submission
    send_to: str | None  # address the user confirmed for emailing the application
    submission: dict[str, Any] | None  # a Submission dump

    # Module 10: Memory
    allow_duplicate: bool  # input: continue even if this job was already submitted
    history: dict[str, Any] | None  # a HistoryCheck dump
    application_id: int | None  # tracker row for this run's outcome

    # Status tracking — reducers let parallel nodes append in the same step
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]

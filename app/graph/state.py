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

    # Module 5: Tailored resume (a TailoredResume dump)
    tailored_resume: dict[str, Any] | None
    tailoring_feedback: list[str]

    # Module 6: Cover letter (a CoverLetter dump)
    cover_letter_tone: str
    cover_letter: dict[str, Any] | None

    # Module 7: Evaluation
    fit_score: float | None
    gaps: list[str]
    improvement_suggestions: list[str]

    # Control flow
    iteration_count: int
    max_iterations: int
    human_approved: bool | None

    # Status tracking — reducers let parallel nodes append in the same step
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]

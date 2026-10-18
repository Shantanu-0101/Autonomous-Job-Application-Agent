import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

from langgraph.types import interrupt

from app.config import settings
from app.evaluator.scorer import assess_fit, documents_to_revise, review_application
from app.generators.cover_letter import TONES, generate_cover_letter
from app.generators.edits import edit_cover_letter, revert_bullets
from app.generators.resume_tailor import tailor_resume
from app.graph.state import AgentState
from app.llm.models import (
    ApplicationReview,
    CompanyResearch,
    CoverLetter,
    FitAssessment,
    HumanDecision,
    JobPosting,
    ParsedResume,
    Submission,
    TailoredResume,
    Tone,
)
from app.memory.history import check_history, record_application
from app.memory.job_memory import get_job_memory
from app.memory.tracker import ApplicationStatus, get_tracker
from app.parsers.resume_parser import (
    ResumeParseError,
    extract_text,
    find_link_issues,
    load_resume_file,
    structure_resume,
)
from app.tools.email_sender import EmailNotConfiguredError, EmailSendError
from app.tools.researcher import research_company
from app.tools.scraper import parse_job_text, scrape_job_posting
from app.tools.submitter import application_url, submit_by_email

logger = logging.getLogger("job_agent.graph")


def _load_inputs(state: AgentState) -> tuple[ParsedResume, JobPosting, CompanyResearch | None] | None:
    resume_data = state.get("parsed_resume")
    job_data = state.get("job_details")
    if not (resume_data and job_data):
        return None
    research_data = state.get("company_research")
    return (
        ParsedResume.model_validate(resume_data),
        JobPosting.model_validate(job_data),
        CompanyResearch.model_validate(research_data) if research_data else None,
    )


def parse_resume_node(state: AgentState) -> dict:
    if state.get("parsed_resume"):
        logger.info("Node: parse_resume — using pre-parsed resume")
        return {"completed_steps": ["parse_resume"]}

    path = state.get("resume_path")
    if not path:
        return {"errors": ["Resume parsing failed: no resume provided"]}

    logger.info("Node: parse_resume — parsing %s", path)
    try:
        data, filename = load_resume_file(path)
        text = extract_text(data, filename)
        parsed = structure_resume(text)
    except ResumeParseError as e:
        logger.warning("Resume parsing failed: %s", e)
        return {"errors": [f"Resume parsing failed: {e}"]}
    except Exception as e:
        logger.exception("Unexpected error while parsing resume")
        return {"errors": [f"Resume parsing failed: {e}"]}

    return {
        "resume_text": text,
        "parsed_resume": parsed.model_dump(),
        "resume_warnings": find_link_issues(text, parsed),
        "completed_steps": ["parse_resume"],
    }


def scrape_job_node(state: AgentState) -> dict:
    description = state.get("job_description")
    url = state.get("job_url", "")
    if not (description or url):
        return {"errors": ["Scraping failed: no job URL or description provided"]}

    try:
        if description:
            logger.info("Node: scrape_job — parsing pasted job description")
            posting = parse_job_text(description, source_url=url)
        else:
            logger.info("Node: scrape_job — scraping %s", url)
            posting = scrape_job_posting(url)
    except Exception as e:
        logger.error("Scraping failed: %s", e)
        return {"errors": [f"Scraping failed: {e}"]}

    return {"job_details": posting.model_dump(), "completed_steps": ["scrape_job"]}


def research_company_node(state: AgentState) -> dict:
    job = state.get("job_details")
    if not job:
        return {"errors": ["Cannot research company: no job details available"]}

    company_name = job.get("company", "")
    if not company_name:
        return {"errors": ["Cannot research company: company name not found in job details"]}

    tracker = get_tracker()
    cached = tracker.cached_research(company_name, settings.research_cache_days)
    if cached:
        logger.info("Node: research_company — using cached research for %s", company_name)
        return {"company_research": cached.model_dump(), "completed_steps": ["research_company"]}

    logger.info("Node: research_company — researching %s", company_name)
    try:
        research = research_company(company_name)
    except Exception as e:
        logger.error("Company research failed: %s", e)
        return {"errors": [f"Company research failed: {e}"]}
    tracker.save_research(company_name, research)
    return {"company_research": research.model_dump(), "completed_steps": ["research_company"]}


def check_history_node(state: AgentState) -> dict:
    job_data = state.get("job_details")
    if not job_data:
        return {}  # scraping failed; the research step reports it
    job = JobPosting.model_validate(job_data)
    logger.info("Node: check_history — %s at %s", job.title, job.company)
    check = check_history(job, get_tracker(), get_job_memory())
    updates: dict[str, Any] = {"history": check.model_dump(), "completed_steps": ["check_history"]}
    if check.blocked and not state.get("allow_duplicate"):
        updates["errors"] = [f"{check.message} Choose 'apply anyway' to prepare a new application."]
    return updates


def tailor_resume_node(state: AgentState) -> dict:
    inputs = _load_inputs(state)
    if inputs is None:
        return {"errors": ["Tailoring skipped: resume or job details are missing"]}

    logger.info("Node: tailor_resume")
    try:
        tailored = tailor_resume(*inputs)
    except Exception as e:
        logger.error("Resume tailoring failed: %s", e)
        return {"errors": [f"Resume tailoring failed: {e}"]}

    return {"tailored_resume": tailored.model_dump(), "completed_steps": ["tailor_resume"]}


def _tone(state: AgentState) -> Tone | None:
    tone = state.get("cover_letter_tone") or "professional"
    return cast(Tone, tone) if tone in TONES else None


def write_cover_letter_node(state: AgentState) -> dict:
    inputs = _load_inputs(state)
    if inputs is None:
        return {"errors": ["Cover letter skipped: resume or job details are missing"]}
    tone = _tone(state)
    if tone is None:
        return {"errors": [f"Cover letter skipped: unknown tone '{state.get('cover_letter_tone')}'"]}

    logger.info("Node: write_cover_letter (%s)", tone)
    try:
        letter = generate_cover_letter(*inputs, tone=tone)
    except Exception as e:
        logger.error("Cover letter generation failed: %s", e)
        return {"errors": [f"Cover letter generation failed: {e}"]}

    return {"cover_letter": letter.model_dump(), "completed_steps": ["write_cover_letter"]}


def assess_fit_node(state: AgentState) -> dict:
    inputs = _load_inputs(state)
    if inputs is None:
        return {"errors": ["Fit assessment skipped: resume or job details are missing"]}

    logger.info("Node: assess_fit")
    resume, job, _ = inputs
    try:
        fit = assess_fit(resume, job)
    except Exception as e:
        logger.error("Fit assessment failed: %s", e)
        return {"errors": [f"Fit assessment failed: {e}"]}

    return {"fit_assessment": fit.model_dump(), "completed_steps": ["assess_fit"]}


def review_application_node(state: AgentState) -> dict:
    inputs = _load_inputs(state)
    candidate_resume = state.get("candidate_tailored_resume")
    candidate_letter = state.get("candidate_cover_letter")
    resume_doc = candidate_resume or state.get("tailored_resume")
    letter_doc = candidate_letter or state.get("cover_letter")
    if inputs is None or not (resume_doc and letter_doc):
        return {"errors": ["Review skipped: tailored resume or cover letter is missing"]}

    resume, job, _ = inputs
    fit_data = state.get("fit_assessment")
    logger.info("Node: review_application (revision %d)", state.get("revision_count", 0))
    try:
        review = review_application(
            resume,
            job,
            FitAssessment.model_validate(fit_data) if fit_data else None,
            TailoredResume.model_validate(resume_doc),
            CoverLetter.model_validate(letter_doc),
        )
    except Exception as e:
        logger.error("Application review failed: %s", e)
        return {"errors": [f"Application review failed: {e}"]}

    # An automatic revision replaces the current document only if it scores at least as well.
    # Revisions the user asked for are always applied — the user, not the reviewer, decides.
    previous_data = state.get("application_review")
    previous = ApplicationReview.model_validate(previous_data) if previous_data else None
    user_requested = bool(state.get("user_requested_revision"))
    if user_requested:
        review.notes.append("Applied the changes you requested.")
    updates: dict[str, Any] = {
        "candidate_tailored_resume": None,
        "candidate_cover_letter": None,
        "user_requested_revision": False,
    }
    if candidate_resume:
        if not user_requested and previous and review.resume_score < previous.resume_score:
            review.resume_score, review.resume_feedback = previous.resume_score, previous.resume_feedback
            review.notes.append("Kept the previous resume: the revision scored lower.")
        else:
            updates["tailored_resume"] = candidate_resume
    if candidate_letter:
        if not user_requested and previous and review.cover_letter_score < previous.cover_letter_score:
            review.cover_letter_score = previous.cover_letter_score
            review.cover_letter_feedback = previous.cover_letter_feedback
            review.notes.append("Kept the previous cover letter: the revision scored lower.")
        else:
            updates["cover_letter"] = candidate_letter

    review.revision = state.get("revision_count", 0)
    review.notes = (previous.notes if previous else []) + review.notes
    updates["application_review"] = review.model_dump()
    updates["completed_steps"] = ["review_application"]
    return updates


def revise_application_node(state: AgentState) -> dict:
    inputs = _load_inputs(state)
    review_data = state.get("application_review")
    tone = _tone(state)
    if inputs is None or not review_data or tone is None:
        return {"errors": ["Revision skipped: missing inputs or review"]}

    # The user's own feedback takes priority over the automatic reviewer's.
    human = state.get("human_feedback")
    if human:
        resume_feedback = human.get("resume", [])
        letter_feedback = human.get("cover_letter", [])
    else:
        review = ApplicationReview.model_validate(review_data)
        docs = documents_to_revise(review, settings.review_quality_threshold)
        resume_feedback = review.resume_feedback if "resume" in docs else []
        letter_feedback = review.cover_letter_feedback if "cover_letter" in docs else []

    revision = state.get("revision_count", 0) + 1
    logger.info(
        "Node: revise_application (revision %d, %s feedback: resume=%d, letter=%d)",
        revision, "user" if human else "reviewer", len(resume_feedback), len(letter_feedback),
    )

    jobs: dict[str, Future] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if resume_feedback:
            jobs["candidate_tailored_resume"] = pool.submit(
                tailor_resume, *inputs, feedback=resume_feedback
            )
        if letter_feedback:
            jobs["candidate_cover_letter"] = pool.submit(
                generate_cover_letter, *inputs, tone=tone, feedback=letter_feedback
            )

    updates: dict[str, Any] = {
        "revision_count": revision,
        "human_feedback": None,
        "user_requested_revision": bool(human),
        "completed_steps": ["revise_application"],
    }
    errors = []
    for key, future in jobs.items():
        try:
            updates[key] = future.result().model_dump()
        except Exception as e:
            logger.error("Revision failed for %s: %s", key, e)
            errors.append(f"Revision failed ({key.removeprefix('candidate_')}): {e}")
    if errors:
        updates["errors"] = errors
    return updates


def human_review_node(state: AgentState) -> dict:
    # The graph pauses here until resumed with a HumanDecision. On resume this node runs again
    # from the top, so nothing before interrupt() may have side effects.
    decision = HumanDecision.model_validate(
        interrupt({"step": "human_review", "revision": state.get("revision_count", 0)})
    )
    logger.info("Node: human_review — user chose %s", decision.action)

    if decision.action == "reject":
        return {"approval_status": "rejected", "completed_steps": ["human_review"]}

    if decision.action == "request_changes":
        return {
            "approval_status": "changes_requested",
            "human_feedback": {
                "resume": [decision.resume_feedback.strip()] if decision.resume_feedback.strip() else [],
                "cover_letter": (
                    [decision.cover_letter_feedback.strip()] if decision.cover_letter_feedback.strip() else []
                ),
            },
            "completed_steps": ["human_review"],
        }

    updates: dict[str, Any] = {
        "approval_status": "approved",
        "send_to": decision.send_to,
        "completed_steps": ["human_review"],
    }
    tailored_data = state.get("tailored_resume")
    if decision.reverted_bullets and tailored_data:
        tailored = revert_bullets(TailoredResume.model_validate(tailored_data), decision.reverted_bullets)
        updates["tailored_resume"] = tailored.model_dump()
    letter_data = state.get("cover_letter")
    if decision.cover_letter_text is not None and letter_data:
        letter = edit_cover_letter(CoverLetter.model_validate(letter_data), decision.cover_letter_text)
        updates["cover_letter"] = letter.model_dump()
    return updates


def submit_application_node(state: AgentState) -> dict:
    existing = state.get("submission")
    if existing and existing.get("status") == "sent":
        return {}  # never send the same application twice

    job = JobPosting.model_validate(state["job_details"])
    send_to = state.get("send_to")
    if not send_to:
        submission = Submission(method="manual", status="ready", apply_url=application_url(job))
        return {"submission": submission.model_dump(), "completed_steps": ["submit_application"]}

    logger.info("Node: submit_application — emailing %s", send_to)
    try:
        submission = submit_by_email(
            send_to,
            job,
            TailoredResume.model_validate(state["tailored_resume"]),
            CoverLetter.model_validate(state["cover_letter"]),
        )
    except (EmailSendError, EmailNotConfiguredError) as e:
        logger.error("Sending failed: %s", e)
        failed = Submission(method="email", status="failed", recipient=send_to, error=str(e))
        return {"submission": failed.model_dump(), "errors": [f"Sending failed: {e}"]}

    return {"submission": submission.model_dump(), "completed_steps": ["submit_application"]}


def record_application_node(state: AgentState) -> dict:
    job_data = state.get("job_details")
    if not job_data:
        return {}
    submission_data = state.get("submission")
    submission = Submission.model_validate(submission_data) if submission_data else None
    if state.get("approval_status") == "rejected":
        status: ApplicationStatus = "skipped"
    elif submission and submission.status == "sent":
        status = "sent"
    else:
        status = "ready"
    fit_data = state.get("fit_assessment")

    try:
        record = record_application(
            JobPosting.model_validate(job_data),
            status,
            get_tracker(),
            get_job_memory(),
            fit_score=fit_data.get("score") if fit_data else None,
            submitted_to=submission.recipient if submission and submission.status == "sent" else "",
        )
    except Exception as e:
        logger.error("Could not record the application: %s", e)
        return {"errors": [f"Could not save this application to your history: {e}"]}
    logger.info("Node: record_application — #%d %s", record.id, status)
    return {"application_id": record.id, "completed_steps": ["record_application"]}

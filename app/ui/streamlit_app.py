import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

import app.logging_config  # noqa: F401  (configures logging)
from app.config import settings
from app.generators.export import (
    cover_letter_docx,
    cover_letter_pdf,
    resume_docx,
    resume_pdf,
    safe_filename,
)
from app.generators.render import render_markdown
from app.graph.builder import build_graph
from app.graph.runner import (
    STEP_LABELS,
    awaiting_approval,
    current_state,
    new_run_config,
    resume_command,
    stream_steps,
)
from app.llm.models import (
    EMAIL_PATTERN,
    ApplicationReview,
    CompanyResearch,
    CoverLetter,
    FitAssessment,
    HumanDecision,
    JobPosting,
    Submission,
    TailoredResume,
)
from app.memory.history import HistoryCheck
from app.memory.tracker import STATUSES, get_tracker
from app.tools.email_sender import EmailSendError, email_configured, verify_email_settings
from app.tools.submitter import application_email

UPLOAD_DIR = Path("data/resumes/uploads")
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# Inputs carried over when re-running a job (e.g. "Apply anyway").
RERUN_KEYS = (
    "parsed_resume", "resume_text", "resume_warnings", "job_url", "job_description",
    "cover_letter_tone", "max_revisions",
)
STATUS_LABELS = {
    "met": ":green[Met]",
    "partial": ":orange[Partial]",
    "not_met": ":red[Missing]",
    "unknown": ":gray[Not on resume]",
}


@st.cache_resource
def get_graph() -> CompiledStateGraph:
    # One graph per server process, shared by all sessions. Checkpoints live in SQLite so runs
    # waiting for approval survive restarts and page refreshes.
    path = Path(settings.checkpoint_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return build_graph(SqliteSaver(sqlite3.connect(path, check_same_thread=False)))


def run_with_progress(payload: dict[str, Any] | Command, config: RunnableConfig, label: str) -> bool:
    with st.status(label, expanded=True) as status:
        try:
            for step in stream_steps(get_graph(), payload, config):
                st.write(f"✓ {STEP_LABELS[step]}")
        except Exception as e:  # surface failures instead of a Streamlit traceback
            status.update(label="Something went wrong", state="error")
            st.error(f"The run stopped unexpectedly: {e}")
            return False
        status.update(label="Done", state="complete", expanded=False)
    return True


def start_run(inputs: dict[str, Any], label: str) -> None:
    config = new_run_config()
    st.session_state["run_config"] = config
    # The run id in the URL lets a refresh (or a bookmark) reopen the same run.
    st.query_params["run"] = config["configurable"]["thread_id"]
    run_with_progress(inputs, config, label)


def start_over() -> None:
    config = st.session_state.pop("run_config", None)
    st.query_params.clear()
    if config:
        # Saved checkpoints include the resume text; don't keep them once the user is done.
        checkpointer = get_graph().checkpointer
        if isinstance(checkpointer, BaseCheckpointSaver):
            checkpointer.delete_thread(config["configurable"]["thread_id"])


# --- Input screen ---


def input_screen() -> None:
    st.subheader("1. Your resume")
    upload = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    st.subheader("2. The job")
    url_tab, text_tab = st.tabs(["Job link", "Paste the description"])
    with url_tab:
        job_url = st.text_input("Job posting URL", placeholder="https://job-boards.greenhouse.io/...")
        st.caption(
            "Greenhouse, Lever, and Ashby links work best. LinkedIn and Naukri block automated "
            "reading — paste the description instead."
        )
    with text_tab:
        job_text = st.text_area("Job description", height=250)

    ready = upload is not None and bool(job_url.strip() or job_text.strip())
    if not st.button("Tailor my application", type="primary", disabled=not ready):
        return
    assert upload is not None

    data = upload.getvalue()
    if len(data) > settings.max_resume_size_mb * 1024 * 1024:
        st.error(f"Resumes must be under {settings.max_resume_size_mb} MB.")
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{Path(upload.name).suffix.lower()}"
    path.write_bytes(data)

    inputs: dict[str, Any] = {
        "resume_path": str(path),
        "job_url": job_url.strip(),
        "cover_letter_tone": st.session_state.get("tone", "professional"),
        "max_revisions": st.session_state.get("max_revisions", settings.max_revisions),
    }
    if job_text.strip():
        inputs["job_description"] = job_text.strip()

    try:
        start_run(inputs, "Working on your application — this takes 1-3 minutes")
    finally:
        path.unlink(missing_ok=True)  # the parsed resume lives in the run state; drop the upload
    st.rerun()


def history_section(state: dict[str, Any]) -> None:
    history_data = state.get("history")
    if not history_data:
        return
    history = HistoryCheck.model_validate(history_data)
    stopped = history.blocked and not state.get("allow_duplicate")
    if stopped:
        st.error(history.message)
        if st.button("Apply anyway"):
            # Reuse the already-parsed resume: the uploaded file was deleted after the first run.
            inputs = {key: state[key] for key in RERUN_KEYS if state.get(key) is not None}
            start_run({**inputs, "allow_duplicate": True}, "Preparing a new application")
            st.rerun()
    elif history.message:
        st.warning(history.message)
    if history.similar:
        with st.expander(f"Similar jobs you've applied to ({len(history.similar)})"):
            for match in history.similar:
                found = match.application
                st.markdown(
                    f"**{found.title}** at {found.company} — {found.status}, "
                    f"{found.updated_at[:10]} · {match.similarity:.0%} similar"
                )


# --- Results ---


def fit_section(state: dict[str, Any]) -> None:
    for warning in state.get("resume_warnings") or []:
        st.error(warning)

    fit_data = state.get("fit_assessment")
    if not fit_data:
        return
    fit = FitAssessment.model_validate(fit_data)
    score_col, text_col = st.columns([1, 3])
    score_col.metric("Fit", "n/a" if fit.score is None else f"{fit.score}/100")
    text_col.markdown(f"**{fit.verdict}**")
    text_col.caption(
        "Based on which requirements your resume shows. Rewriting can't change this score — "
        "only real experience can."
    )
    for warning in fit.warnings:
        st.warning(warning)
    with st.expander("Requirement checklist"):
        for r in fit.requirements:
            optional = " *(nice to have)*" if r.importance == "nice_to_have" else ""
            st.markdown(f"{STATUS_LABELS[r.status]} — {r.requirement}{optional}")
            if r.evidence:
                st.caption(r.evidence)


def resume_tab(tailored: TailoredResume) -> None:
    st.markdown(render_markdown(tailored.resume))
    if tailored.matched_keywords:
        st.markdown("**Matched:** " + ", ".join(tailored.matched_keywords))
    if tailored.missing_keywords:
        st.markdown("**Missing (not added to your resume):** " + ", ".join(tailored.missing_keywords))
    if tailored.warnings:
        with st.expander(f"Fact-check notes ({len(tailored.warnings)})"):
            for w in tailored.warnings:
                st.caption(w)


def changes_tab(tailored: TailoredResume, revision: int) -> list[int]:
    if not tailored.bullet_changes:
        st.info("No bullets were rewritten.")
        return []
    st.caption("Tick any rewrite you'd rather undo. Your original wording will be used instead.")
    reverted = []
    for i, change in enumerate(tailored.bullet_changes):
        before, after = st.columns(2)
        before.markdown(f"**Before**\n\n{change.original}")
        after.markdown(f"**After**\n\n{change.rewritten}")
        if st.checkbox("Keep my original", key=f"revert_{revision}_{i}"):
            reverted.append(i)
        st.divider()
    return reverted


def letter_tab(letter: CoverLetter, revision: int) -> str:
    st.caption(f"{letter.greeting} … {letter.sign_off} {letter.signature}")
    text = st.text_area(
        "Cover letter (edit freely — paragraphs are separated by blank lines)",
        value="\n\n".join(letter.paragraphs),
        height=400,
        key=f"letter_{revision}",
    )
    for w in letter.warnings:
        st.warning(w)
    return text


def job_tab(state: dict[str, Any]) -> None:
    job = JobPosting.model_validate(state["job_details"])
    st.markdown(f"**{job.title}** at **{job.company}** — {job.location or 'Location not stated'}")
    st.caption(f"Salary: {job.salary_range}")
    for title, items in (
        ("Requirements", job.requirements),
        ("Nice to have", job.nice_to_have),
        ("Responsibilities", job.responsibilities),
    ):
        if items:
            st.markdown(f"**{title}**\n" + "\n".join(f"- {i}" for i in items))
    research_data = state.get("company_research")
    if research_data:
        research = CompanyResearch.model_validate(research_data)
        st.markdown(f"**About {research.name}**\n\n{research.summary}")
        if research.recent_news:
            st.markdown("**Recent news**\n" + "\n".join(f"- {n}" for n in research.recent_news))


def send_option(job: JobPosting) -> tuple[str | None, bool]:
    """Returns (recipient to email, whether approving is allowed). Sending is always opt-in."""
    address = application_email(job)
    if not address:
        return None, True
    if not email_configured():
        st.info(
            f"This job takes applications by email at {address}. Download your documents after "
            "approving and email them yourself, or set the SMTP_* values in .env to send from here."
        )
        return None, True

    if not st.checkbox(f"Also email my application to {address}", key="send_email"):
        return None, True
    recipient = st.text_input(
        "Recipient (check it against the job posting)", value=address, key="send_to"
    ).strip()
    if not EMAIL_PATTERN.fullmatch(recipient):
        st.error("Enter a valid email address.")
        return None, False
    st.caption(
        f"Your resume and cover letter PDFs will be sent from {settings.sender_email} to {recipient}. "
        "This can't be undone."
    )
    return recipient, True


def decide(decision: HumanDecision, config: RunnableConfig, label: str) -> None:
    if run_with_progress(resume_command(decision), config, label):
        st.rerun()


def review_screen(state: dict[str, Any], config: RunnableConfig) -> None:
    tailored = TailoredResume.model_validate(state["tailored_resume"])
    letter = CoverLetter.model_validate(state["cover_letter"])
    review = ApplicationReview.model_validate(state["application_review"])
    revision = state.get("revision_count", 0)

    st.caption(
        f"AI review — resume {review.resume_score}/100, cover letter {review.cover_letter_score}/100"
        + (f" · {revision} revision(s)" if revision else "")
    )
    for note in review.notes:
        st.caption(f"• {note}")

    resume_t, changes_t, letter_t, job_t = st.tabs(
        ["Tailored resume", f"Changes ({len(tailored.bullet_changes)})", "Cover letter", "Job & company"]
    )
    with resume_t:
        resume_tab(tailored)
    with changes_t:
        reverted = changes_tab(tailored, revision)
    with letter_t:
        letter_text = letter_tab(letter, revision)
    with job_t:
        job_tab(state)

    st.subheader("Your decision")
    send_to, send_ok = send_option(JobPosting.model_validate(state["job_details"]))
    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve and send" if send_to else "Approve", type="primary", disabled=not send_ok):
        decide(
            HumanDecision(
                action="approve",
                reverted_bullets=reverted,
                cover_letter_text=letter_text,
                send_to=send_to,
            ),
            config,
            "Sending your application" if send_to else "Saving your approval",
        )
    if reject_col.button("Reject — don't apply"):
        decide(HumanDecision(action="reject"), config, "Recording your decision")

    with st.form("request_changes"):
        st.markdown("**Or ask for changes**")
        resume_fb = st.text_area("What should change in the resume?", height=80)
        letter_fb = st.text_area("What should change in the cover letter?", height=80)
        if st.form_submit_button("Revise"):
            if not (resume_fb.strip() or letter_fb.strip()):
                st.error("Describe at least one change.")
            else:
                decide(
                    HumanDecision(
                        action="request_changes",
                        resume_feedback=resume_fb,
                        cover_letter_feedback=letter_fb,
                    ),
                    config,
                    "Revising — this takes about a minute",
                )


def finished_screen(state: dict[str, Any]) -> None:
    status = state.get("approval_status")
    if status == "rejected":
        st.info("You chose not to apply to this job.")
        return
    if status != "approved":
        return

    submission_data = state.get("submission")
    submission = Submission.model_validate(submission_data) if submission_data else None
    if submission and submission.status == "sent":
        st.success(f"Sent to {submission.recipient}. A copy of your documents is below.")
    elif submission and submission.status == "failed":
        st.error(f"Approved, but the email wasn't sent: {submission.error}")
        st.caption("Download your documents below and send them yourself.")
    else:
        st.success("Approved. Download your documents and apply on the job's page.")
        if submission and submission.apply_url:
            st.link_button("Open the application page", submission.apply_url)

    tailored = TailoredResume.model_validate(state["tailored_resume"])
    letter = CoverLetter.model_validate(state["cover_letter"])
    contact = tailored.resume.contact
    name = safe_filename(contact.name) or "Applicant"
    resume_col, letter_col = st.columns(2)
    resume_col.download_button(
        "Resume (PDF)", resume_pdf(tailored.resume), f"{name}_Resume.pdf", "application/pdf"
    )
    resume_col.download_button("Resume (Word)", resume_docx(tailored.resume), f"{name}_Resume.docx", DOCX)
    letter_col.download_button(
        "Cover letter (PDF)", cover_letter_pdf(letter, contact), f"{name}_Cover_Letter.pdf",
        "application/pdf",
    )
    letter_col.download_button(
        "Cover letter (Word)", cover_letter_docx(letter, contact), f"{name}_Cover_Letter.docx", DOCX
    )


def results_screen(config: RunnableConfig) -> None:
    graph = get_graph()
    state = current_state(graph, config)
    if not state:
        st.warning("This session has expired. Please start over.")
        return

    job = state.get("job_details")
    if job:
        st.header(f"{job['title']} at {job['company']}")
    history_section(state)
    fit_section(state)

    if awaiting_approval(graph, config):
        review_screen(state, config)
    elif state.get("approval_status"):
        finished_screen(state)
    history_message = (state.get("history") or {}).get("message") or ""
    for error in state.get("errors") or []:
        if not (history_message and error.startswith(history_message)):  # already shown above
            st.error(error)


# --- Application history ---


def applications_screen() -> None:
    tracker = get_tracker()
    records = tracker.list()
    if not records:
        st.info("No applications yet. Approved and skipped jobs will appear here.")
        return

    counts = {s: sum(r.status == s for r in records) for s in ("sent", "applied", "interviewing", "offer")}
    for column, (status, count) in zip(st.columns(len(counts)), counts.items()):
        column.metric(status.capitalize(), count)

    st.caption("Update the status and notes as you hear back. Changes save automatically.")
    table = pd.DataFrame([
        {
            "id": r.id, "Company": r.company, "Role": r.title, "Status": r.status,
            "Fit": r.fit_score, "Updated": r.updated_at[:10], "Notes": r.notes, "Link": r.job_url or None,
        }
        for r in records
    ])
    edited = st.data_editor(
        table,
        key="applications_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["id", "Company", "Role", "Fit", "Updated", "Link"],
        column_config={
            "id": None,
            "Status": st.column_config.SelectboxColumn(options=list(STATUSES), required=True),
            "Link": st.column_config.LinkColumn(display_text="Open"),
        },
    )
    for before, after in zip(table.to_dict("records"), edited.to_dict("records")):
        if (before["Status"], before["Notes"]) != (after["Status"], after["Notes"]):
            tracker.update_status(int(after["id"]), after["Status"], notes=after["Notes"] or "")


# --- Page ---


def main() -> None:
    st.set_page_config(page_title="Job Application Agent", layout="wide")
    st.title("Job Application Agent")
    st.caption(
        "Tailors your resume and cover letter to a job using only facts from your resume. "
        "Nothing is sent without your approval."
    )

    # Reopen the run named in the URL after a refresh or restart.
    run_id = st.query_params.get("run")
    if "run_config" not in st.session_state and run_id:
        st.session_state["run_config"] = {"configurable": {"thread_id": run_id}}

    # In the main area, not the sidebar: the sidebar collapses on narrow screens.
    view = st.radio(
        "View", ["New application", "My applications"], key="view",
        horizontal=True, label_visibility="collapsed",
    )

    with st.sidebar:
        st.header("Settings")
        st.selectbox("Cover letter tone", ["professional", "enthusiastic", "concise"], key="tone")
        st.slider(
            "Automatic revision rounds", 0, 3, value=settings.max_revisions, key="max_revisions",
            help="More rounds can improve the documents but take longer and use more of your LLM quota.",
        )
        st.caption(f"LLM provider: {settings.llm_provider}")
        if email_configured() and st.button("Check email settings"):
            try:
                verify_email_settings()
            except EmailSendError as e:
                st.error(str(e))
            else:
                st.success(f"Signed in to {settings.smtp_host} as {settings.smtp_username}. Nothing was sent.")
        if st.session_state.get("run_config") and st.button("Start over"):
            start_over()
            st.rerun()

    if view == "My applications":
        applications_screen()
        return
    config = st.session_state.get("run_config")
    if config is None:
        input_screen()
    else:
        results_screen(config)

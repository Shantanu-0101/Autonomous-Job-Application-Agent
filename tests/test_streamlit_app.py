import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import settings
from app.graph.builder import build_graph
from app.graph.runner import current_state, new_run_config
from app.memory.tracker import get_tracker
from app.llm.models import (
    ApplicationReview,
    BulletChange,
    CompanyResearch,
    ContactInfo,
    CoverLetter,
    Experience,
    FitAssessment,
    JobPosting,
    ParsedResume,
    RequirementMatch,
    Submission,
    TailoredResume,
)

RESUME = ParsedResume(
    contact=ContactInfo(name="Jane Doe", email="jane@example.com"),
    experience=[Experience(title="Engineer", company="Acme", bullets=["Built APIs"])],
)
TAILORED = TailoredResume(
    resume=RESUME.model_copy(update={
        "experience": [Experience(title="Engineer", company="Acme", bullets=["Designed APIs"])]
    }),
    bullet_changes=[BulletChange(
        section="experience", entry=0, position=0, original="Built APIs", rewritten="Designed APIs"
    )],
    matched_keywords=["APIs"],
)
LETTER = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["I build APIs."], sign_off="Best,",
                     signature="Jane Doe")
FIT = FitAssessment(score=70, verdict="Good match", requirements=[
    RequirementMatch(requirement="Python", importance="required", status="met", evidence="Built APIs"),
])
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
REVIEW = ApplicationReview(resume_score=90, resume_feedback=[], cover_letter_score=90, cover_letter_feedback=[])


def job(**update) -> JobPosting:
    return JobPosting(title="Backend Engineer", company="Globex", description="Build APIs",
                      source_url="https://boards.io/globex/1", **update)


@pytest.fixture
def paused_app(request):
    """A Streamlit app session looking at a run (normally paused for approval), LLM steps mocked.

    Parametrize with a JobPosting, or a dict {"job": ..., "already_sent": True} to seed history.
    """
    param = getattr(request, "param", None)
    options = param if isinstance(param, dict) else {"job": param or job()}
    posting = options["job"]
    if options.get("already_sent"):
        get_tracker().record(posting, "sent")
    graph = build_graph()
    patches = {
        "app.graph.nodes.load_resume_file": (b"data", "r.pdf"),
        "app.graph.nodes.extract_text": "resume text",
        "app.graph.nodes.structure_resume": RESUME,
        "app.graph.nodes.scrape_job_posting": posting,
        "app.graph.nodes.research_company": CompanyResearch(name="Globex", summary="Makes APIs."),
        "app.graph.nodes.assess_fit": FIT,
        "app.graph.nodes.review_application": REVIEW,
        "app.graph.nodes.tailor_resume": TAILORED,
        "app.graph.nodes.generate_cover_letter": LETTER,
        "app.ui.streamlit_app.get_graph": graph,
    }
    with contextlib.ExitStack() as stack:
        mocks = {t: stack.enter_context(patch(t, return_value=v)) for t, v in patches.items()}
        config = new_run_config()
        graph.invoke({"resume_path": "r.pdf", "job_url": posting.source_url}, config)
        at = AppTest.from_file(APP, default_timeout=30)
        at.session_state["run_config"] = config
        at.run()
        yield at, mocks


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_input_screen_waits_for_resume_and_job():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert button(at, "Tailor my application").disabled


def test_review_screen_shows_fit_changes_and_letter(paused_app):
    at, _ = paused_app
    assert not at.exception
    assert at.header[0].value == "Backend Engineer at Globex"
    assert at.metric[0].value == "70/100"
    assert [t.label for t in at.tabs] == ["Tailored resume", "Changes (1)", "Cover letter", "Job & company"]
    assert at.text_area[0].value == "I build APIs."
    assert any(c.label == "Keep my original" for c in at.checkbox)
    assert button(at, "Approve").label == "Approve"


def test_approve_shows_downloads_and_apply_link(paused_app):
    at, _ = paused_app
    button(at, "Approve").click().run()

    assert not at.exception
    assert at.success[0].value == "Approved. Download your documents and apply on the job's page."
    assert not at.error


def test_undoing_a_rewrite_and_editing_the_letter(paused_app):
    at, mocks = paused_app
    next(c for c in at.checkbox if c.label == "Keep my original").check()
    at.text_area[0].input("I build fast APIs.")
    button(at, "Approve").click().run()

    assert not at.exception
    graph = mocks["app.ui.streamlit_app.get_graph"].return_value
    state = current_state(graph, at.session_state["run_config"])
    assert TailoredResume.model_validate(state["tailored_resume"]).resume.experience[0].bullets == ["Built APIs"]
    assert CoverLetter.model_validate(state["cover_letter"]).paragraphs == ["I build fast APIs."]


@pytest.mark.parametrize("paused_app", [job(application_method="jobs@globex.io")], indirect=True)
def test_email_jobs_offer_opt_in_sending(paused_app, monkeypatch):
    at, _ = paused_app
    for name, value in {"smtp_host": "smtp.x.com", "smtp_username": "me@x.com",
                        "smtp_password": "pw", "sender_email": "me@x.com"}.items():
        monkeypatch.setattr(settings, name, value)
    at.run()

    send_box = next(c for c in at.checkbox if c.label.startswith("Also email my application"))
    assert send_box.label == "Also email my application to jobs@globex.io"
    send_box.check().run()
    assert at.text_input[0].value == "jobs@globex.io"

    sent = Submission(method="email", status="sent", recipient="jobs@globex.io")
    with patch("app.graph.nodes.submit_by_email", return_value=sent) as mock_submit:
        button(at, "Approve and send").click().run()

    assert not at.exception
    assert mock_submit.call_args.args[0] == "jobs@globex.io"
    assert at.success[0].value == "Sent to jobs@globex.io. A copy of your documents is below."


@pytest.mark.parametrize("paused_app", [job(application_method="jobs@globex.io")], indirect=True)
def test_invalid_recipient_blocks_approval(paused_app, monkeypatch):
    at, _ = paused_app
    for name, value in {"smtp_host": "smtp.x.com", "smtp_username": "me@x.com",
                        "smtp_password": "pw", "sender_email": "me@x.com"}.items():
        monkeypatch.setattr(settings, name, value)
    at.run()
    next(c for c in at.checkbox if c.label.startswith("Also email")).check().run()
    at.text_input[0].input("jobs at globex").run()

    assert at.error[0].value == "Enter a valid email address."
    assert button(at, "Approve").disabled


def test_request_changes_revises_and_returns_to_review(paused_app):
    at, mocks = paused_app
    at.text_area[1].input("Lead with Python")
    button(at, "Revise").click().run()

    assert not at.exception
    assert mocks["app.graph.nodes.tailor_resume"].call_args.kwargs == {"feedback": ["Lead with Python"]}
    assert any("Applied the changes you requested." in c.value for c in at.caption)


def test_reject(paused_app):
    at, _ = paused_app
    button(at, "Reject — don't apply").click().run()
    assert at.info[0].value == "You chose not to apply to this job."


# --- Memory ---


def test_approved_application_appears_in_my_applications(paused_app):
    at, _ = paused_app
    button(at, "Approve").click().run()
    at.radio(key="view").set_value("My applications").run()

    assert not at.exception
    table = at.dataframe[0].value if at.dataframe else None
    assert table is not None
    row = table.iloc[0]
    assert (row["Company"], row["Role"], row["Status"], row["Fit"]) == ("Globex", "Backend Engineer", "ready", 70)
    assert [m.label for m in at.metric] == ["Sent", "Applied", "Interviewing", "Offer"]


def test_empty_history_page():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["view"] = "My applications"
    at.run()
    assert at.info[0].value.startswith("No applications yet.")


@pytest.mark.parametrize("paused_app", [{"job": job(), "already_sent": True}], indirect=True)
def test_already_submitted_job_offers_apply_anyway(paused_app):
    at, mocks = paused_app
    assert at.error[0].value.startswith("You already applied to Backend Engineer at Globex")
    assert len(at.error) == 1  # the blocked message isn't repeated in the error list
    assert mocks["app.graph.nodes.tailor_resume"].call_count == 0

    button(at, "Apply anyway").click().run()

    assert not at.exception
    assert mocks["app.graph.nodes.tailor_resume"].call_count == 1
    assert mocks["app.graph.nodes.load_resume_file"].call_count == 1  # reused the parsed resume
    assert any(b.label == "Approve" for b in at.button)


def test_run_reopens_from_the_url(paused_app):
    at, mocks = paused_app
    run_id = at.session_state["run_config"]["configurable"]["thread_id"]

    reopened = AppTest.from_file(APP, default_timeout=30)
    reopened.query_params["run"] = run_id
    reopened.run()

    assert not reopened.exception
    assert reopened.header[0].value == "Backend Engineer at Globex"
    assert any(b.label == "Approve" for b in reopened.button)


def test_start_over_deletes_the_saved_run(paused_app):
    at, mocks = paused_app
    graph = mocks["app.ui.streamlit_app.get_graph"].return_value
    config = at.session_state["run_config"]

    next(b for b in at.sidebar.button if b.label == "Start over").click().run()

    assert "run_config" not in at.session_state
    assert current_state(graph, config) == {}
    assert button(at, "Tailor my application").disabled


# --- Access ---


def test_password_gate_hides_everything_until_correct(monkeypatch):
    monkeypatch.setattr(settings, "app_access_password", "s3cret")
    monkeypatch.setattr("app.ui.streamlit_app.time.sleep", lambda _s: None)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not any(b.label == "Tailor my application" for b in at.button)

    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert at.error[0].value == "Wrong password."
    assert not any(b.label == "Tailor my application" for b in at.button)

    at.text_input[0].input("s3cret")
    at.button[0].click().run()
    assert any(b.label == "Tailor my application" for b in at.button)


def test_no_password_configured_means_open_locally():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any(b.label == "Tailor my application" for b in at.button)
